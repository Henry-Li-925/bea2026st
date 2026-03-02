import torch
import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import XLMRobertaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

class VAEClassificationHead(nn.Module):
    def __init__(self, config, latent_dim=64):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
        # 1. Encoder: Maps to Mean (mu) and Log-Variance (logvar)
        self.mu = nn.Linear(config.hidden_size, latent_dim)
        self.logvar = nn.Linear(config.hidden_size, latent_dim)
        
        # 2. Classifier: Maps latent Z -> Labels
        self.decoder_classifier = nn.Linear(latent_dim, config.num_labels)

    def reparameterize(self, mu, logvar):
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

    def forward(self, features):
        x = features[:, 0, :] # Extract <s> token
        x = self.activation(self.dense(x))
        
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        z = self.reparameterize(mu, logvar)
        logits = self.decoder_classifier(z)
        
        return logits, mu, logvar

class XLMRobertaVAE(XLMRobertaForSequenceClassification):
    def __init__(self, config, latent_dim = 64, beta = 1e-3):
        super().__init__(config)
        # Initialize custom VAE head
        self.classifier = VAEClassificationHead(config, latent_dim=latent_dim)
        
        # Weight for KL Divergence Loss (Hyperparameter)
        # Controls trade-off between accuracy and regularization
        self.beta = beta
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # --- FIX: Clean kwargs before passing to backbone ---
        # The Trainer passes 'num_items_in_batch' which the backbone doesn't accept.
        kwargs.pop("num_items_in_batch", None)

        # 1. Run the Backbone (Encoder)
        outputs = self.roberta(input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        
        # 2. Run the VAE Head
        logits, mu, logvar = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # A. Calculate Prediction Loss (MSE or CrossEntropy)
            if self.config.problem_type == "regression" or self.num_labels == 1:
                loss_fct = MSELoss(reduction='sum')
                pred_loss = loss_fct(logits.squeeze(), labels.squeeze()) / input_ids.size(0)
            else:
                loss_fct = CrossEntropyLoss(reduction='sum')
                pred_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) / input_ids.size(0)
            
            # B. Calculate KL Divergence Loss
            # Measures how much our latent distribution deviates from a standard normal distribution
            # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Normalize KL by batch size to keep scale consistent
            kl_loss = kl_loss / input_ids.size(0)
            
            # C. Total Loss
            loss = pred_loss + (self.beta * kl_loss)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )