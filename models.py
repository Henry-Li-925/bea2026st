import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.modeling_outputs import ModelOutput

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CustomClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    token_attn_weights: Optional[torch.FloatTensor] = None

class VIBClassificationHead(nn.Module):
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
        x = self.activation(self.dense(features))
        
        mu = self.mu(x)
        logvar = self.logvar(x)
        
        z = self.reparameterize(mu, logvar)
        logits = self.decoder_classifier(z)
        
        return logits, mu, logvar

class ScalarMix(nn.Module):
    def __init__(
        self, 
        mixture_size: int=13, 
        do_layer_norm: bool = True,
        initial_scalar_parameters: List[float] = None,
        have_temperature: bool = True,
        trainable: bool = False
        ) -> None:
        """
        Layer mixing through global scalar weights. Adapted from [https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py].
        Important implemenation strategy:
            - do_layer_norm: perform layer normalization for numerical stability.
            - have_temperature: introduce an additional temp parameter that regulates the scalar weights. tau = base_temp + factor * temp
            - trainable: whether the gamma parameter is trainable.
        """
        
        super().__init__()

        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = torch.zeros((mixture_size,)) # zero initialization
        
        self.softmax = nn.Softmax(0)

        if not trainable:
            self.register_buffer('gamma', torch.tensor([1.0])) # If not trainable, Gamma is fixed
        else:
            self.gamma = nn.Parameter(torch.tensor([1.0]))


        self.have_temperature = have_temperature
        if self.have_temperature:
            # adding a new learnable parameter: temperature
            self.base_temp = 1e2
            self.factor = 1e5
            self.temp = nn.Parameter(torch.tensor([1e-2]))
            
        self.scalar_weights = nn.Parameter(
            initial_scalar_parameters
            )
    
    def forward(
        self, 
        tensors: List[torch.Tensor],
        mask: torch.bool=None
        ) -> torch.Tensor:
        """
        Values:
            tensors = [layer_1, layer_2, ..., layer_n]
            weights = [w_1, ..., w_n]
        Formula:
            tau = base_temp + factor * temp
            normed_weights = layer_norm(weights * tau)
            output = gamma * sum(softmax(normed_weights) * tensors)
        Return:
            a batched tensor of the shape (batch_size, seq_len, hidden_size)
        """
        # gamma * sum(softmax(weight) * tensor, dim=1)

        if self.have_temperature:
            # tau = base_temp + factor * temp
            # weight = scalar_weight * tau
            tau = self.base_temp + self.factor*self.temp
            normed_weights = self.softmax(self.scalar_weights * tau) # (mixture_size)
        else:
            normed_weights = self.softmax(self.scalar_weights) # (mixture_size)
        
        # Elmo-style global layer norm. Adapted from AllenNLP's Scalar Mix.
        def _Elmo_do_layer_norm(tensors, broadcast_mask, num_elements_not_masked, eps = 1e-13):
            # (x - mean) / sqrt(variance + eps)
            masked_pt = broadcast_mask * tensors # (n_layer, batch_size, seq_len, hidden_size)
            num_elements_not_masked = num_elements_not_masked[None, :, None] # (1, batch_size, 1)
            mean = torch.sum(masked_pt, dim=-2) / num_elements_not_masked # (n_layer, batch_size, hidden_size)
            variance = torch.sum(((masked_pt - mean.unsqueeze(-2)) * broadcast_mask) **2, dim=-2) / num_elements_not_masked # (n_layer, batch_size, hidden_size)
            return (masked_pt - mean.unsqueeze(-2)) / torch.sqrt(variance.unsqueeze(-2) + eps) # (n_layer, batch_size, seq_len, hidden_size)

        mixed_tensor = None
        if self.do_layer_norm:
            assert mask is not None
            broadcast_mask = mask[None, :, :, None] # (1, batch_size, seq_len, 1)
            
            # Token-wise layer norm
            tensors = torch.stack(tensors, 0) # (n_layer, batch_size, seq_len, hidden_size)
            input_dim = tensors.size(-1) # hidden_size for each sequence.

            # PyTorch's provides off-the-shelf layer_norm function that is readily vectorized.  
            normed_tensors = F.layer_norm(tensors, normalized_shape=(input_dim, )) # tokenwise normalization: normalize across the last dimension only.
            # seq_len = tensors.size(-2) 
            # normed_tensors = F.layer_norm(tensors, normalized_shape=(seq_len, input_dim)) # sequencewise normalization: normalize across the last two dimensions
            normed_tensors = normed_tensors * broadcast_mask # masking normed tensors
            
            weights_shape = (-1,) + (1,) * (tensors.dim() - 1) # (-1, 1, 1, 1)
            normed_weights = normed_weights.view(weights_shape) # broadcastable to tensors
            mixed_tensor = torch.sum(normed_weights * normed_tensors, dim=0) # (1, batch_size, seq_len, hidden_size)

        else:
            # vectorized for efficiency
            tensors = torch.stack(tensors, 0) # (layer, batch_size, seq_len, hidden_dim)
            weights_shape = (-1,) + (1,) * (tensors.dim() - 1) # (-1, 1, 1, 1)
            normed_weights = normed_weights.view(weights_shape) # broadcastable to tensors
            mixed_tensor = torch.sum(normed_weights * tensors, dim=0) # (batch_size, seq_len, hidden_size)
        
        return self.gamma * mixed_tensor

class PositionalEncoding(nn.Module): 
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
        
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=128, latent_dim=768):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_seq_len, latent_dim)
    
    def forward(self, x):
        positions = torch.arange(x.size(-2)).expand(x.size(0), -1).to(x.device)
        embed = self.positional_embedding(positions)
        return x + embed

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        # Projects hidden states to compute attention scores
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        # 1. Compute unnormalized attention scores
        # Shape: [batch_size, seq_len, 1] -> [batch_size, seq_len]
        scores = self.v(torch.tanh(self.W(hidden_states))).squeeze(-1)
        
        # 2. Mask out padding tokens so they don't receive attention
        # We use a large negative number (-1e9) so softmax outputs 0
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # 3. Normalize scores into probabilities (alpha weights)
        attn_weights = torch.softmax(scores, dim=-1) # [batch_size, seq_len]
        
        # 4. Compute the weighted sum of hidden states (Context Vector)
        # unsqueeze attn_weights to [batch, 1, seq_len] for batch matrix multiplication
        # output shape: [batch, 1, hidden_size] -> [batch, 1, hidden_size]
        context_vector = torch.bmm(self.dropout(attn_weights).unsqueeze(1), hidden_states)
        
        return context_vector, attn_weights

class Backbone(PreTrainedModel):
    config_class = AutoConfig
    def __init__(
        self,
        config
    ):
        """
        Instantiate the model backbone as a subclass of PreTrainedModel.
        Due to transformer's design pattern, we have to assign model prefix for backbones of different structures. 
        e.g. roberta: ["xlm-roberta", "roberta"]
        """
        super().__init__(config)
        self.config = config
        model_type = config.model_type
        if model_type in ['xlm-roberta', 'roberta']:
            # add_pooling_layer=False to get raw hidden states
            self.roberta = AutoModel.from_config(config, add_pooling_layer=False)
            self.__class__.base_model_prefix = "roberta"
    
    def forward(self):
        raise NotImplementedError("ERROR: forward not implemented!")

class CloseTrack_Predictor(Backbone):
    def __init__(
        self,
        config,
        dropout: float = 0.1,
        latent_dim: int = 64,
        beta: float = 1e-3, 
        layer_wise: Optional[str] = None,
        token_wise: Optional[str] = None,
        pred_head: str = 'mlp',
        **kwargs
    ):
        """
        Our customized predictor that performs layer fusion and/or token fusion strategies to exploit transformer's internal structure. Currently, we have implemented 
        one method for layer fusion (layer_wise) and two for token fusion (token_wise). If both are not provided, the predictor reduces to the ordinary regressor, 
        using the first token from the last hidden state as predictor. 
            - layer_wise:
                - "ScalarMix": the layer fusion method that learns a global set of weights for each internal layers before pooling into a single hidden state. 
            - token_wise: 
                - 'AddAttn': the Additive Attention (see [https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html#additive-attention])
                - 'SelfAttn': the Multi-head self-attention, implemented by the PyTorch off-the-self nn.MultiheadAttention function.
                    Since the Positional Encoding are used in transformer training, two types of positional encoder are used for self-attention for investigation
                    - Sinusodial: [https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html#positional-encoding]
                    - Learned: [https://machinelearningmastery.com/positional-encodings-in-transformer-models/]
            - pred_head: the regressor used for prediction
                - 'mlp': the same MLP regressor as used in XLMRobertaForSequenceClassification
                - 'vib': a testing regressor that follows the idea of Variational Information Bottleneck, which is essentially a regularized regressor.
        """
        super().__init__(config)

        # initiate layer fusion strategy
        self.layer_wise = layer_wise

        if self.layer_wise == 'ScalarMix':  
            self.ScalarMix = ScalarMix(
                config.num_hidden_layers + 1
                )

        # initiate token fusion strategy
        self.token_wise = token_wise

        if self.token_wise == "AddAttn":
            self.attention  = AdditiveAttention(
                hidden_size = config.hidden_size,
                dropout = dropout
                )

        elif self.token_wise == "SelfAttn":
            self.attention = nn.modules.activation.MultiheadAttention(
                embed_dim = config.hidden_size,
                num_heads = 1,
                dropout = dropout,
                batch_first = True
                )
        
        # inititate Positional Encoder
        self.pos_encoding = kwargs.get('pos_encoding', False)
        self.learned_pos = kwargs.get('learned_pos', False)
        self.max_seq_len = kwargs.get('max_seq_len', 128)

        if self.pos_encoding:
            if self.learned_pos:
                self.pos_encoder = LearnedPositionalEncoding(
                    max_seq_len=self.max_seq_len, 
                    latent_dim=config.hidden_size
                    )
            else:
                self.pos_encoder = PositionalEncoding(
                    num_hiddens=config.hidden_size, 
                    dropout=dropout, 
                    max_len=self.max_seq_len
                    )

        # initiate regressor
        self.beta = beta
        self.pred_head = pred_head
        if self.pred_head == 'mlp':
            self.regressor = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Dropout(dropout),
                nn.Linear(config.hidden_size, config.num_labels)
            )
        elif self.pred_head == 'vib':
            self.regressor = VIBClassificationHead(
                config, 
                latent_dim=latent_dim
                )

        # Initialize weights (Hugging Face standard)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        **kwargs
        ) -> CustomClassifierOutput:
        """
        Argument:
            input_ids: the tokenized input 
            attnention_mask: the sequence mask output by the tokenizer
            labels: the target labels
        Return:
            CustomClassifierOutput(
                loss: prediction loss
                logits: predicted labels
                hidden_states: all hidden states generated by the backbone
                attentions: all attention scores calculated by the backbone
                token_attn_weights: the token-level attention weights calculated by the token fusion method.
            )
        """

        # The Trainer passes 'num_items_in_batch' which the backbone doesn't accept.
        if "num_items_in_batch" not in self.config:
            kwargs.pop("num_items_in_batch", None)

        # identify the backbone and generate the hidden states. 
        if hasattr(self, "roberta"):
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

        attn_weights = None

        # if layer fusion is applied
        if self.layer_wise:
            # mixes the entire sequences from all layers, not just the first token to support following token-wise aggregation
            hiddens = list(outputs.hidden_states) # [layer, batch, seq, hidden]

            if self.layer_wise == "ScalarMix":
                hiddens = self.ScalarMix(hiddens, mask = attention_mask) # [1, batch, seq, hidden]
            else:
                # reduce to the last hidden states rn, wait for other layer fusion methods 
                hiddens = hiddens[-1] # [batch, seq, hidden]
        else:
            # else, uses only the last hidden states.
            hiddens = outputs.last_hidden_state # [batch, seq, hidden]

        # if token fusion is applied
        if self.token_wise:
            if self.token_wise == "AddAttn":
                hiddens, attn_weights = self.attention(hiddens, attention_mask) # hiddens: [batch, 1, hidden_size] attn_weights: [batch, seq]
            elif self.token_wise == "SelfAttn":
                padding_masks = ~attention_mask.to(torch.bool) # key_padding_mask argument in MultiheadAttention uses boolean mask where True indicates padding.
                # if positional encoding is applied
                if self.pos_encoding:
                    hiddens = self.pos_encoder(hiddens)
                
                hiddens, attn_weights = self.attention(hiddens, hiddens, hiddens, key_padding_mask  = padding_masks) # hiddens: [batch, seq, hidden_size] attn_weights: [batch, seq, seq]

                # extract only </s> token's attention weights
                attn_weights = attn_weights[:,0,:] # [batch_size, seq_len, seq_len] -> [batch_size, seq_len]

        # predict with the first token   
        hiddens = hiddens[:, 0, :] # [batch, 1, hidden]
        
        if self.pred_head == 'mlp':
            logits = self.regressor(hiddens) 
            loss = None           
            if labels is not None:
                loss_fct = nn.MSELoss(reduction='mean')
                loss = loss_fct(logits.squeeze(), labels.squeeze())

        elif self.pred_head == 'vib':
            logits, mu, logvar = self.regressor(hiddens)
            loss = None
            if labels is not None:     
                loss_fct = nn.MSELoss(reduction='sum')
                pred_loss = loss_fct(logits.squeeze(), labels.squeeze()) / input_ids.size(0)

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / input_ids.size(0)

                loss = pred_loss + (self.beta * kl_loss)
        
        return CustomClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_attn_weights=attn_weights
        )

class Ensemble_MLP(PreTrainedModel):
    config_class = AutoConfig
    def __init__(
        self, 
        config, 
        dropout: float = 0.1,
        full_sequence: bool = False,
        l1_aug: bool = True
        ):

        super().__init__(config)
        
        self.config = config
        self.full_sequence = full_sequence

        model_type = config.model_type

        if model_type in ['xlm-roberta', 'roberta']:
            # add_pooling_layer=False to get raw hidden states
            self.roberta = AutoModel.from_config(config, add_pooling_layer=False)
            self.__class__.base_model_prefix = "roberta"
        
        # full_sequence = False, use only <CLS> or <s> token for classification
        # no need for layer norm if not full_sequence
        do_layer_norm = full_sequence

        self.ScalarMix = ScalarMix(
            config.num_hidden_layers + 1,
            do_layer_norm = do_layer_norm
            )

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, config.num_labels)
        )

            
        # Initialize weights (Hugging Face standard)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        l1_embed = None,
        **kwargs) -> CustomClassifierOutput:
        # The Trainer passes 'num_items_in_batch' which the backbone doesn't accept.
        if "num_items_in_batch" not in self.config:
            kwargs.pop("num_items_in_batch", None)

        if hasattr(self, "roberta"):
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

        if not self.full_sequence:
            token = [layer[:, 0, :] for layer in outputs.hidden_states]
            mixed_token = self.ScalarMix(token) # (batch, hidden)
        else:
            seq_tokens = [layer for layer in outputs.hidden_states]
            mixed_seq = self.ScalarMix(seq_tokens, mask = attention_mask) # (batch, seq_len, hidden)
            
            # perform mean pooling if using full sequence 
            boardcast_mask = attention_mask.unsqueeze(-1).float() # (batch, seq_len, 1); float type for numerical stability
            sum_embed = torch.sum(mixed_seq * boardcast_mask, dim=1)
            sum_mask = torch.sum(boardcast_mask, dim=1).clamp(min=1e-9) # clamp to avoid zero-division due to all padding
            mixed_token = sum_embed/sum_mask # --> (batch, hidden)


        logits = self.mlp(mixed_token)   

        loss = None
        if labels is not None:     
            loss_fct = nn.MSELoss(reduction='mean')
            loss = loss_fct(logits.squeeze(), labels.squeeze())
            
        return CustomClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_attn_weights=None
        )

class Ensemble_MLP_vib(PreTrainedModel):
    config_class = AutoConfig
    def __init__(
        self, 
        config, 
        latent_dim = 64, 
        beta: float = 1e-3, 
        full_sequence: bool = False
        ):

        super().__init__(config)
        
        self.config = config
        self.full_sequence = full_sequence

        model_type = config.model_type

        if model_type in ['xlm-roberta', 'roberta']:
            # add_pooling_layer=False to get raw hidden states
            self.roberta = AutoModel.from_config(config, add_pooling_layer=False)
            self.__class__.base_model_prefix = "roberta"
        
        # full_sequence = False, use only <CLS> or <s> token for classification
        # no need for layer norm if not full_sequence
        do_layer_norm = full_sequence

        self.ScalarMix = ScalarMix(
            config.num_hidden_layers + 1,
            do_layer_norm = do_layer_norm
            )

        self.vib = VIBClassificationHead(config, latent_dim=latent_dim)
        self.beta = beta

        # Initialize weights (Hugging Face standard)
        self.post_init()

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        **kwargs) -> CustomClassifierOutput:
        # The Trainer passes 'num_items_in_batch' which the backbone doesn't accept.
        if "num_items_in_batch" not in self.config:
            kwargs.pop("num_items_in_batch", None)

        if hasattr(self, "roberta"):
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

        if not self.full_sequence:
            token = [layer[:, 0, :] for layer in outputs.hidden_states]
            mixed_token = self.ScalarMix(token) # (batch, hidden)
        else:
            seq_tokens = [layer for layer in outputs.hidden_states]
            mixed_seq = self.ScalarMix(seq_tokens, mask = attention_mask) # (batch, seq_len, hidden)
            
            # perform mean pooling if using full sequence 
            boardcast_mask = attention_mask.unsqueeze(-1).float() # (batch, seq_len, 1); float type for numerical stability
            sum_embed = torch.sum(mixed_seq * boardcast_mask, dim=1)
            sum_mask = torch.sum(boardcast_mask, dim=1).clamp(min=1e-9) # clamp to avoid zero-division due to all padding
            mixed_token = sum_embed/sum_mask # --> (batch, hidden)

        logits, mu, logvar = self.vib(mixed_token)
        
        loss = None
        if labels is not None:     
            loss_fct = nn.MSELoss(reduction='sum')
            pred_loss = loss_fct(logits.squeeze(), labels.squeeze()) / mixed_token.size(0)

            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / mixed_token.size(0)

            loss = pred_loss + (self.beta * kl_loss)
            
        return CustomClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_attn_weights=None
        )
