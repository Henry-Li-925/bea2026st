import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import XLMRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from dataclasses import dataclass, field
from transformers import DataCollatorWithPadding
from typing import Optional, Union, List, Dict, Any


class CVAEhead(nn.Module):
    def __init__(self, config, latent_size, class_size):
        super().__init__()
        self.feature_size = config.hidden_size
        self.class_size = class_size
        self.one_hot = F.one_hot

        self.fc1 = nn.Linear(config.hidden_size + class_size, 400)
        self.mu = nn.Linear(400, latent_size)
        self.logvar = nn.Linear(400, latent_size)

        # decode
        self.fc2 = nn.Linear(latent_size + class_size, 400)
        self.fc3 = nn.Linear(400, config.hidden_size)

        # regressor
        self.regressor = nn.Sequential(
            nn.Linear(latent_size + class_size, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

        self.elu = nn.ELU()
        self.gelu = nn.GELU()

    def encode(self, x, c):
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        input = torch.cat([x, c], 1)
        h1 = self.elu(self.fc1(input))
        mu = self.mu(h1)
        logvar = self.logvar(h1)

        return mu, logvar

    def reparametrize(self, mu, logvar):
        """
        The "Reparameterization Trick":
        z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, we just use the mean (deterministic)
            return mu
    
    def decode(self, z, c):
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        input = torch.cat([z, c], 1)
        h2 = self.elu(self.fc2(input))
        output = self.gelu(self.fc3(h2))
        return output

    def forward(self, x, c):
        c_one_hot = self.one_hot(c.long(), self.class_size).float()

        # CVAE recon
        mu, logvar = self.encode(x.view(-1, self.feature_size), c_one_hot)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decode(z, c_one_hot)

        # Latent pred
        input = torch.cat([z, c_one_hot], 1)
        y_pred = self.regressor(input)
        return y_pred, x_recon, mu, logvar

class CVIBhead(nn.Module):
    def __init__(self, config, latent_size, language_class_size):
        super().__init__()
        self.feature_size = config.hidden_size
        self.c_size = language_class_size
        self.one_hot = F.one_hot
        
        # Inference Network: q(z | x, y, c)
        # Note: In a true regression CVAE, y is included in the encoder during training
        self.fc_enc = nn.Linear(self.feature_size + self.c_size + 1, 400)
        self.mu = nn.Linear(400, latent_size)
        self.logvar = nn.Linear(400, latent_size)

        # Generative Network: p(y | x, z, c)
        self.fc_dec1 = nn.Linear(latent_size + self.c_size, 400)
        self.fc_dec2 = nn.Linear(400, 1) # Predicts the continuous difficulty score y

        self.elu = nn.ELU()

    def encode(self, x, c, y):
        # Concatenate text features (x), language features (c), and target (y)
        inputs = torch.cat([x, c, y.view(-1, 1)], dim=1)
        h1 = self.elu(self.fc_enc(inputs))
        return self.mu(h1), self.logvar(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference, we just use the mean (deterministic)
            return mu
    
    def decode(self, z, c):
        # Decoder uses x, z, and c to predict y
        inputs = torch.cat([z, c], dim=1)
        h2 = self.elu(self.fc_dec1(inputs))
        y_hat = self.fc_dec2(h2) # Linear output for continuous score
        return y_hat


    def forward(self, x, c, y=None):
        x = x.view(-1, self.feature_size)
        c_one_hot = self.one_hot(c.long(), self.c_size).float()

        mu, logvar = self.encode(x, c_one_hot, y)
        z = self.reparameterize(mu, logvar)
            
        y_recon = self.decode(z, c_one_hot)
        return y_recon, mu, logvar

class XLMRobertaCVAE(XLMRobertaForSequenceClassification):
    def __init__(self, config, latent_dim = 64, beta = 1e-3, class_size = 3):
        super().__init__(config)
        # Initialize custom VAE head
        self.classifier = CVAEhead(config, latent_size=latent_dim, class_size = class_size)
        
        # Weight for KL Divergence Loss (Hyperparameter)
        # Controls trade-off between accuracy and regularization
        self.beta = beta
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, l1_encode=None, **kwargs):
        # --- FIX: Clean kwargs before passing to backbone ---
        # The Trainer passes 'num_items_in_batch' which the backbone doesn't accept.
        kwargs.pop("num_items_in_batch", None)

        # 1. Run the Backbone (Encoder)
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        output_token = sequence_output[:, 0, :] # <s> token

        y_pred, x_recon, mu, logvar = self.classifier(output_token, l1_encode)
        
        loss = None
        if labels is not None:
            loss_func = MSELoss(reduction='sum')
            y_loss = loss_func(y_pred.squeeze(-1), labels.squeeze(-1)) / input_ids.size(0)
            x_loss = loss_func(x_recon, output_token) / input_ids.size(0)

            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input_ids.size(0)

            loss = (y_loss + x_loss) + (self.beta * kld)
    
        return SequenceClassifierOutput(
                loss=loss,
                logits=y_pred,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

class XLMRobertaCVIB(XLMRobertaForSequenceClassification):
    def __init__(self, config, latent_dim = 64, beta = 1e-3, class_size = 3):
        super().__init__(config)
        # Initialize custom VAE head
        self.classifier = CVIBhead(config, latent_size=latent_dim, language_class_size = class_size)
        
        # Weight for KL Divergence Loss (Hyperparameter)
        # Controls trade-off between accuracy and regularization
        self.beta = beta
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, l1_encode=None, **kwargs):
        # --- FIX: Clean kwargs before passing to backbone ---
        # The Trainer passes 'num_items_in_batch' which the backbone doesn't accept.
        kwargs.pop("num_items_in_batch", None)

        # 1. Run the Backbone (Encoder)
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        output_token = sequence_output[:, 0, :] # <s> token

        y_recon, mu, logvar = self.classifier(output_token, l1_encode, y=labels)
        
        loss = None
        if labels is not None:
            loss_func = MSELoss(reduction='sum')
            pred_loss = loss_func(y_recon.squeeze(-1), labels.squeeze(-1)) / input_ids.size(0)

            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input_ids.size(0)

            loss = pred_loss + (self.beta * kld)
    
        return SequenceClassifierOutput(
                loss=loss,
                logits=y_recon,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

@dataclass
class CVAEDataCollator(DataCollatorWithPadding):
    """
    Data collator that inherits all arguments from DataCollatorWithPadding, 
    with an added argument to specify custom features (e.g., regression labels, categorical IDs)
    that should be safely extracted before sequence padding.
    """
    # New argument to dynamically specify non-text features
    # Using field(default_factory=...) ensures mutable defaults are handled safely in dataclasses
    custom_features: List[str] = field(default_factory=lambda: ["l1_encode", "labels"])

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Initialize a dictionary to hold the extracted non-text features
        extracted_features = {key: [] for key in self.custom_features}
        
        # 2. Pop the custom features out of the dictionaries so the tokenizer doesn't try to pad them
        for feature in features:
            for key in self.custom_features:
                if key in feature:
                    extracted_features[key].append(feature.pop(key))

        # 3. Pass the remaining text features (input_ids, attention_mask) to the parent class for standard padding
        batch = super().__call__(features)

        # 4. Add the custom features back to the batch as correctly typed PyTorch tensors
        if "l1_encode" in extracted_features and extracted_features["l1_encode"]:
            # Categorical variables (for F.one_hot) strictly require torch.long
            batch["l1_encode"] = torch.tensor(extracted_features["l1_encode"], dtype=torch.long)
        
        if "labels" in extracted_features and extracted_features["labels"]:
            # Regression targets require torch.float32
            batch["labels"] = torch.tensor(extracted_features["labels"], dtype=torch.float32)
            
        # Optional: Handle any other dynamic features the user might add later
        for key in self.custom_features:
            if key not in ["l1_encode", "labels"] and extracted_features.get(key):
                batch[key] = torch.tensor(extracted_features[key], dtype=torch.float32)

        return batch