import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, AutoConfig, DataCollatorWithPadding
from transformers.modeling_outputs import ModelOutput

import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict


@dataclass
class CustomClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    token_attn_weights: Optional[torch.FloatTensor] = None
    
@dataclass
class MultiTaskClassifierOutput(CustomClassifierOutput):
    """Extends output to include auxiliary POS logits and loss."""
    pos_logits: torch.FloatTensor = None

@dataclass
class MultiTaskDataCollator(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract POS labels if they exist
        pos_labels = [f.pop("pos_labels") for f in features] if "pos_labels" in features[0] and features[0]["pos_labels"] is not None else None
        
        # 2. Safely remove standard 'labels' if they are None to prevent the base class crash
        for f in features:
            if "labels" in f and f["labels"] is None:
                del f["labels"]
            if "label" in f and f["label"] is None:
                del f["label"]
        
        # Base class handles input_ids, masks, and regression 'labels'
        batch = super().__call__(features)

        if pos_labels is not None:
            # Simple tensor conversion (Batch_Size,)
            batch["pos_labels"] = torch.tensor(pos_labels, dtype=torch.long)

        return batch

# Not Used
class VIBHead(nn.Module):
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

class MaxPooling(nn.Module):
    def __init__(
        self,
        dim:int=0,
        cls_only:bool=True,
        **kwargs
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cls_only = cls_only
    
    def forward(self, x, **kwargs):
        if self.cls_only:
            x = [layer[:, 0, :] for layer in x]
            x = torch.stack(x, self.dim).unsqueeze(-2)
        else:
            x = torch.stack(x, self.dim)
        return torch.max(x, self.dim).values

class MeanPooling(nn.Module):
    def __init__(
        self,
        dim:int=0,
        cls_only:bool=True
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cls_only = cls_only
        
    def forward(self, x, **kwargs):
        if self.cls_only:
            x = [layer[:, 0, :] for layer in x]
            x = torch.stack(x, self.dim).unsqueeze(-2)
        else:
            x = torch.stack(x, self.dim)
        return torch.mean(x, self.dim)

class ScalarMix(nn.Module):
    def __init__(
        self, 
        mixture_size: int=13, 
        do_layer_norm: bool = True,
        initial_scalar_parameters: List[float] = None,
        trainable: bool = True
        ) -> None:
        """
        Layer mixing through global scalar weights. Adapted from [https://github.com/allenai/allennlp/blob/main/allennlp/modules/scalar_mix.py].
        Important implemenation strategy:
            - do_layer_norm: perform layer normalization for numerical stability.
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

        self.scalar_weights = nn.Parameter(
            initial_scalar_parameters
            )
    
    def forward(
        self, 
        tensors: List[torch.Tensor],
        mask: torch.bool=None,
        **kwargs
        ) -> torch.Tensor:
        """
        Values:
            tensors = [layer_1, layer_2, ..., layer_n]
            weights = [w_1, ..., w_n]
        Formula:
            normed_weights = layer_norm(weights)
            output = gamma * sum(softmax(normed_weights) * tensors)
        Return:
            a batched tensor of the shape (batch_size, seq_len, hidden_size)
        """

        normed_weights = self.softmax(self.scalar_weights) # (mixture_size)
        
        mixed_tensor = None
        if self.do_layer_norm:
            assert mask is not None
            broadcast_mask = mask[None, :, :, None] # (1, batch_size, seq_len, 1)
            
            # Token-wise layer norm
            tensors = torch.stack(tensors, 0) # (n_layer, batch_size, seq_len, hidden_size)
            
            # PyTorch's provides off-the-shelf layer_norm function that is readily vectorized.  
            normed_tensors = F.layer_norm(tensors, normalized_shape=(tensors.size(-1), )) # tokenwise normalization: normalize across the last dimension only.
            # seq_len = tensors.size(-2) 
            # normed_tensors = F.layer_norm(tensors, normalized_shape=(tensors.size(-2), tensors.size(-1))) # sequencewise normalization: normalize across the last two dimensions
            normed_tensors = normed_tensors * broadcast_mask # masking normed tensors
            
            weights_shape = (-1,) + (1,) * (tensors.dim() - 1) # (-1, 1, 1, 1)
            normed_weights = normed_weights.view(weights_shape) # broadcastable to tensors
            mixed_tensor = torch.sum(normed_weights * normed_tensors, dim=0) # (1, batch_size, seq_len, hidden_size)

        else:
            # if no layer norm, by default we extract only the first token
            tensors = [layer[:, 0, :] for layer in tensors] # extract the first token (batch_size, seq_len, hidden_size) -> (batch_size, hidden_dim)
            tensors = torch.stack(tensors, 0) # (layer, batch_size, hidden_dim)
            tensors = F.layer_norm(tensors, normalized_shape=(tensors.size(-1),)) # normalize within token to prevent gradient explode.
            weights_shape = (-1,) + (1,) * (tensors.dim() - 1) # (-1, 1, 1)
            normed_weights = normed_weights.view(weights_shape) # broadcastable to tensors
            mixed_tensor = torch.sum(normed_weights * tensors, dim=0).unsqueeze(1) # (batch_size, 1, hidden_size)
        
        return self.gamma * mixed_tensor

class CustomModel(PreTrainedModel):
    # 1. Tell Hugging Face which configuration class to use
    config_class = AutoConfig
    
    @property
    def all_tied_weights_keys(self):
        """
        Overrides the internal HF tied weights tracker.
        This model is a regression/classification architecture, 
        so it does not use tied weights (unlike language models).
        """
        return {}
    
    @property
    def backbone(self):
        if hasattr(self, 'roberta'):
            return self.roberta
        elif hasattr(self, 'model'):
            return self.model
        raise AttributeError("Backbone not initialized properly.")
    
    
    def __init__(self,config,**kwargs):
        """
        Our customized predictor that performs layer fusion and/or token fusion strategies to exploit transformer's internal structure. Currently, we have implemented 
        one method for layer fusion (layer_wise) and two for token fusion (token_wise). If both are not provided, the predictor reduces to the ordinary regressor, 
        using the first token from the last hidden state as predictor. 
            - layer_wise:
                - "ScalarMix": the layer fusion method that learns a global set of weights for each internal layers before pooling into a single hidden state. 
            - token_wise: 
                - 'SelfAttn': the Multi-head self-attention, implemented by the PyTorch off-the-self nn.MultiheadAttention function.
            - pred_head: the regressor used for prediction
                - 'mlp': the same MLP regressor as used in XLMRobertaForSequenceClassification
                - 'vib': a testing regressor that follows the idea of Variational Information Bottleneck, which is essentially a regularized regressor.
        """
        super().__init__(config)
        
        # --- CONFIGURATION BINDING ---
        # We save any custom arguments into the config. 
        # This guarantees they are permanently written to `config.json` when you save the model!
        for k, v in kwargs.items():
            setattr(config, k, v)
            
        self.pred_head = getattr(config, 'pred_head', 'mlp')
        self.token_pool = getattr(config, 'token_pool', 'cls')
        self.dropout = getattr(config, 'dropout', 0.1)
        
        model_type = config.model_type
        
        # --- BUILD THE SKELETON (NO WEIGHTS LOADED HERE) ---
        if model_type in ['xlm-roberta', 'roberta']:
            self.base_model_prefix = "roberta"
            self.roberta = AutoModel.from_config(config, 
                                                 add_pooling_layer=False    # Disable XLMRobertaPool which
                                                                            # use cls pooling by default
                                                                            # it is also the default option
                                                                            # when calling XLMRobertaForSequenceClassification
                                                 )
        elif model_type in ['modernbert']:
            self.base_model_prefix = "model"
            self.model = AutoModel.from_config(config)
        else:
            raise NotImplementedError(f"Backbone '{model_type}' is not supported by this predictor.")

         # --- BUILD THE REGRESSION HEAD ---
        hidden_size = config.hidden_size
        
        if self.pred_head == 'mlp':
            if model_type in ['xlm-roberta', 'roberta']:
                # class XLMRobertaClassificationHead(nn.Module):
                # https://github.com/huggingface/transformers/blob/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L918
                self.regressor = nn.Sequential(
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_size, 1)
                )
            elif model_type in ['modernbert']:
                # Following the setup of ModernBERT classifier
                # https://github.com/huggingface/transformers/blob/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/modernbert/modeling_modernbert.py#L493
                # https://github.com/huggingface/transformers/blob/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/modernbert/modeling_modernbert.py#L581
                self.regressor = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size, bias=config.classifier_bias),
                    nn.GELU(),
                    nn.LayerNorm(hidden_size, eps=config.norm_eps, bias=config.norm_bias),
                    nn.Dropout(self.dropout),
                    nn.Linear(hidden_size, 1)
                    )          
        else: # default regressor
            self.regressor = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(hidden_size, 1)
            )
        
        self.layer_pool = getattr(config, 'layer_pool', None)
        self.last_k_layer = getattr(config, 'last_k_layer', None)
        if self.layer_pool:
            if self.layer_pool == 'scalarmix':
                self.layer_pooler = ScalarMix(
                    self.last_k_layer if self.last_k_layer is not None else config.num_hidden_layers + 1,
                    do_layer_norm= self.layer_pool != 'cls',
                    trainable=True
                )
            elif self.layer_pool == 'max':
                self.layer_pooler = MaxPooling(
                    dim = 0,
                    cls_only = self.layer_pool == 'cls'
                )
            elif self.layer_pool == 'mean':
                self.layer_pooler = MeanPooling(
                    dim = 0,
                    cls_only = self.layer_pool == 'cls'
                )

        # initiate token fusion strategy
        self.token_pool = getattr(config, 'token_pool', 'cls')
        if self.token_pool == 'mha':
            if getattr(config, 'num_heads', None) is None:
                raise ValueError(f'The number of multiple heads is not defined!')
            if config.hidden_size % config.num_heads != 0:
                raise ValueError(f"The hidden size ({config.hidden_size}) must be a multiple of num_heads ({config.num_heads}).")
            self.token_pooler = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=config.num_heads,
                dropout=self.dropout,
                batch_first=True
            )
        else:
            self.token_pooler = None

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        **kwargs
    ) -> CustomClassifierOutput:
        """
        Argument:
            input_ids: the tokenized input 
            attention_mask: the sequence mask output by the tokenizer
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

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        if self.layer_pool:
            hiddens = list(outputs.hidden_states)[-self.last_k_layer:] if self.last_k_layer is not None else list(outputs.hidden_states)
            output_hiddens = self.layer_pooler(hiddens, mask=attention_mask)
        else:
            output_hiddens = outputs.last_hidden_state

        attn_weights = None
        
        if self.token_pool == 'mean':
            if attention_mask is None:
                attention_mask = torch.ones(
                    output_hiddens.shape[:2], device=output_hiddens.device, dtype=torch.bool
                )
            mask_float = attention_mask.to(output_hiddens.dtype)
            pooled_output = (output_hiddens * mask_float.unsqueeze(-1)).sum(dim=1) / torch.clamp(mask_float.sum(
                dim=1, keepdim=True
            ), min=1e-9) # prevent division by zero
        elif self.token_pool == 'mha':
            last_cls_hidden = output_hiddens[:, 0, :].unsqueeze(1)
            padding_masks = ~attention_mask.to(torch.bool)
            attn_output, ave_attn_weights = self.token_pooler(
                query=last_cls_hidden,
                key=output_hiddens,
                value=output_hiddens,
                key_padding_mask=padding_masks,
                need_weights=True,
                average_attn_weights=True
            )
            pooled_output = attn_output[:, 0, :]
            attn_weights = ave_attn_weights[:,0,:]
        else:
            pooled_output = output_hiddens[:, 0, :]
            
            
        loss = None
        logits = self.regressor(pooled_output)
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))      
        
        return CustomClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_attn_weights=attn_weights
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        **kwargs
    ) -> CustomClassifierOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        if self.layer_pool:
            hiddens = list(outputs.hidden_states)[-self.last_k_layer:] if self.last_k_layer is not None else list(outputs.hidden_states)
            output_hiddens = self.layer_pooler(hiddens, mask=attention_mask)
        else:
            output_hiddens = outputs.last_hidden_state

        pooled_output, attn_weights = self._perform_token_pooling(output_hiddens, 
                                                                  attention_mask, 
                                                                  self.token_pool, 
                                                                  self.token_pooler
                                                                  )
        
        loss = None
        if self.pred_head == 'vib':        
            logits, mu, logvar = self.regressor(pooled_output)
            
            if labels is not None:
                loss_fct = nn.MSELoss(reduction='sum')
                pred_loss = loss_fct(logits.view(-1), labels.view(-1)) / pooled_output.size(0)

                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / pooled_output.size(0)

                loss = pred_loss + (self.beta * kl_loss)
                
        else:
            logits = self.regressor(pooled_output)
            
            if labels is not None:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))      
                
        return CustomClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_attn_weights=attn_weights
        )
    
    @staticmethod
    def _perform_token_pooling(hiddens, mask, pool_type, pooler_module):
        """Helper to ensure logic parity between paths."""
        # Default CLS
        hiddens_out = hiddens[:, 0, :]
        attn_weights = None
        if pool_type == 'mean':
            m_float = mask.to(hiddens.dtype)
            hiddens_out = (hiddens * m_float.unsqueeze(-1)).sum(dim=1) / torch.clamp(m_float.sum(dim=1, keepdim=True), min=1e-4)

        elif pool_type == 'mha' and pooler_module:
            cls_h = hiddens[:, 0, :].unsqueeze(1)
            p_mask = ~mask.to(torch.bool)
            attn_out, ave_attn_weights = pooler_module(
                query=cls_h, 
                key=hiddens, 
                value=hiddens, 
                key_padding_mask=p_mask, 
                need_weights=True,
                average_attn_weights=True
                )
            hiddens_out = attn_out[:, 0, :]
            attn_weights = ave_attn_weights[:,0,:]
            
        return hiddens_out, attn_weights
        
        
class MultiTaskCustomModel(CustomModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        
        # --- AUXILIARY POS HEAD ---
        self.num_pos_labels = getattr(config, 'num_pos_labels', 7) 
        
        # Classification head for the single POS label
        self.pos_classifier = copy.deepcopy(self.regressor)
        
        # same but separate layer_pooler and token_pooler:
        if hasattr(self, 'token_pooler') and self.token_pooler is not None:
            self.pos_token_pooler = copy.deepcopy(self.token_pooler)
        else:
            self.pos_token_pooler = None
            
        if hasattr(self, 'layer_pooler') and self.layer_pooler is not None:
            self.pos_layer_pooler = copy.deepcopy(self.layer_pooler)
        else:
            self.pos_layer_pooler = None
        
        if isinstance(self.pos_classifier, nn.Sequential):
            last_layer_idx = len(self.pos_classifier) - 1
            in_features = self.pos_classifier[last_layer_idx].in_features
            self.pos_classifier[last_layer_idx] = nn.Linear(in_features, self.num_pos_labels)
        else:
            # Fallback if regressor is a single Linear layer
            in_features = self.pos_classifier.in_features
            self.pos_classifier = nn.Linear(in_features, self.num_pos_labels)
            
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,      # Primary Task (Difficulty - Float)
        pos_labels=None,  # Auxiliary Task (POS - Long/Int)
        **kwargs
    ) -> MultiTaskClassifierOutput:
        
        # Use your established backbone + pooling logic
        # This gives us the pooled_output used for regression
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )
        
        # --- PATH A: PRIMARY REGRESSION (Difficulty) ---
        # 1. Layer Pooling
        if self.layer_pool:
            h_reg = list(outputs.hidden_states)[-self.last_k_layer:] if self.last_k_layer else list(outputs.hidden_states)
            out_h_reg = self.layer_pooler(h_reg, mask=attention_mask)
        else:
            out_h_reg = outputs.last_hidden_state

        # 2. Token Pooling
        p_out_reg, primary_token_attn_weights = self._perform_token_pooling(out_h_reg, attention_mask, self.token_pool, self.token_pooler)
        logits = self.regressor(p_out_reg)

        # --- PATH B: AUXILIARY CLASSIFICATION (POS) ---
        # 1. Layer Pooling (Independent Weights)
        if self.pos_layer_pooler:
            h_pos = list(outputs.hidden_states)[-self.last_k_layer:] if self.last_k_layer else list(outputs.hidden_states)
            out_h_pos = self.pos_layer_pooler(h_pos, mask=attention_mask)
        else:
            out_h_pos = outputs.last_hidden_state

        # 2. Token Pooling (Independent Weights)
        p_out_pos, _ = self._perform_token_pooling(out_h_pos, attention_mask, self.token_pool, self.pos_token_pooler)
        pos_logits = self.pos_classifier(p_out_pos)
            
        primary_loss = None
        pos_loss = None

        # 1. Primary Regression Loss
        if labels is not None:
            loss_fct_reg = nn.MSELoss(reduction='mean')
            primary_loss = loss_fct_reg(logits.view(-1), labels.view(-1))

        # 2. Auxiliary Classification Loss (Single Label per sequence)
        if pos_labels is not None:
            loss_fct_pos = nn.CrossEntropyLoss(reduction='mean')
            pos_loss = loss_fct_pos(pos_logits.view(-1, self.num_pos_labels), pos_labels.view(-1))
            
        total_loss = None
        if primary_loss is not None and pos_loss is not None:
            safe_log_var0 = torch.clamp(self.log_vars[0], min=-10.0, max=10.0)
            safe_log_var1 = torch.clamp(self.log_vars[1], min=-10.0, max=10.0)
            loss1 = torch.exp(-safe_log_var0) * primary_loss + safe_log_var0
            loss2 = torch.exp(-safe_log_var1) * pos_loss + safe_log_var1
            total_loss = loss1 + loss2
            
        return MultiTaskClassifierOutput(
            loss=total_loss,
            logits=logits,
            pos_logits=pos_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_attn_weights=primary_token_attn_weights
        )

class MultiTaskCascadeCustomModel(CustomModel):
    def __init__(self, config, **kwargs):
        # 1. Initialize the Base Model (Builds the backbone and standard regressor)
        super().__init__(config, **kwargs)
        
        self.num_pos_labels = getattr(config, 'num_pos_labels', 20) # Ensure this matches your dataset!
        hidden_size = config.hidden_size
        
        # ==========================================
        # 2. BUILD THE INDEPENDENT POS PATH
        # ==========================================
        self.pos_classifier = copy.deepcopy(self.regressor)
        
        # Deepcopy poolers so they have independent weights
        self.pos_token_pooler = copy.deepcopy(self.token_pooler) if hasattr(self, 'token_pooler') and self.token_pooler else None
        self.pos_layer_pooler = copy.deepcopy(self.layer_pooler) if hasattr(self, 'layer_pooler') and self.layer_pooler else None

        # Fix the POS classifier's output dimension (from 1 to num_pos_labels)
        if isinstance(self.pos_classifier, nn.Sequential):
            last_idx = len(self.pos_classifier) - 1
            in_features = self.pos_classifier[last_idx].in_features
            self.pos_classifier[last_idx] = nn.Linear(in_features, self.num_pos_labels)
        elif hasattr(self.pos_classifier, 'dense'): # Catch VIB Head
            in_features = self.pos_classifier.decoder.in_features
            self.pos_classifier.decoder = nn.Linear(in_features, self.num_pos_labels)
        else:
            in_features = self.pos_classifier.in_features
            self.pos_classifier = nn.Linear(in_features, self.num_pos_labels)

        # ==========================================
        # 3. WIDEN THE PRIMARY REGRESSOR (FEATURE INJECTION)
        # ==========================================
        combined_dim = hidden_size + self.num_pos_labels
        
        if isinstance(self.regressor, nn.Sequential):
            # Find the very first Linear layer and widen its input dimension
            for i, module in enumerate(self.regressor):
                if isinstance(module, nn.Linear):
                    out_features = module.out_features
                    self.regressor[i] = nn.Linear(combined_dim, out_features, bias=module.bias is not None)
                    break
        elif hasattr(self.regressor, 'dense'): # Catch VIB Head
            self.regressor.dense = nn.Linear(combined_dim, hidden_size)
        else:
            out_features = self.regressor.out_features
            self.regressor = nn.Linear(combined_dim, out_features)

        # 4. Learned Log-Variances for Dynamic Precision Weighting
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,      # Primary Task (Difficulty)
        pos_labels=None,  # Auxiliary Task (POS)
        **kwargs
    ) -> MultiTaskClassifierOutput:
        
        # --- THE SHARED BACKBONE (Runs Only Once for Speed/Memory) ---
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # --- PATH A: AUXILIARY CLASSIFICATION (Runs First) ---
        if self.pos_layer_pooler:
            h_pos = list(outputs.hidden_states)[-self.last_k_layer:] if self.last_k_layer else list(outputs.hidden_states)
            out_h_pos = self.pos_layer_pooler(h_pos, mask=attention_mask)
        else:
            out_h_pos = outputs.last_hidden_state

        p_out_pos, _ = self._perform_token_pooling(out_h_pos, attention_mask, self.token_pool, self.pos_token_pooler)
        
        # Handle VIB vs Standard
        if self.pred_head == 'vib':
            pos_logits, _, _ = self.pos_classifier(p_out_pos)
        else:
            pos_logits = self.pos_classifier(p_out_pos)


        # --- PATH B: PRIMARY REGRESSION (The Cascade) ---
        if self.layer_pool:
            h_reg = list(outputs.hidden_states)[-self.last_k_layer:] if self.last_k_layer else list(outputs.hidden_states)
            out_h_reg = self.layer_pooler(h_reg, mask=attention_mask)
        else:
            out_h_reg = outputs.last_hidden_state

        p_out_reg, primary_token_attn_weights = self._perform_token_pooling(out_h_reg, attention_mask, self.token_pool, self.token_pooler)
        
        # 🟢 FEATURE INJECTION MAGIC 
        # Combine the semantic backbone features with the explicit POS knowledge.
        # .detach() ensures the primary loss doesn't ruin the POS classifier's weights.
        combined_features = torch.cat([p_out_reg, pos_logits.detach()], dim=-1)

        if self.pred_head == 'vib':
            logits, mu, logvar = self.regressor(combined_features)
        else:
            logits = self.regressor(combined_features)
            
        # --- LOSS CALCULATION ---
        primary_loss = None
        if labels is not None:
            if self.pred_head == 'vib':
                pred_loss = nn.MSELoss(reduction='sum')(logits.view(-1), labels.view(-1)) / p_out_reg.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / p_out_reg.size(0)
                primary_loss = pred_loss + (self.beta * kl_loss)
            else:
                primary_loss = nn.MSELoss(reduction='mean')(logits.view(-1), labels.view(-1))

        pos_loss = None
        if pos_labels is not None:
            # ignore_index=-1 safely handles null POS tags without crashing CUDA
            pos_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)(
                pos_logits.view(-1, self.num_pos_labels), pos_labels.view(-1)
            )

        # --- DYNAMIC PRECISION WEIGHTING ---
        total_loss = None
        if primary_loss is not None and pos_loss is not None:
            # 🟢 NaN PROTECTION: Clamp exponents to safe FP16 boundaries
            safe_log_var0 = torch.clamp(self.log_vars[0], min=-10.0, max=10.0)
            safe_log_var1 = torch.clamp(self.log_vars[1], min=-10.0, max=10.0)

            loss1 = torch.exp(-safe_log_var0) * primary_loss + safe_log_var0
            loss2 = torch.exp(-safe_log_var1) * pos_loss + safe_log_var1
            total_loss = loss1 + loss2
            
        return MultiTaskClassifierOutput(
            loss=total_loss,
            logits=logits,
            pos_logits=pos_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            token_attn_weights=primary_token_attn_weights
        )