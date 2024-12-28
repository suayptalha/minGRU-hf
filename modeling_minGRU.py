import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional
from .configuration_minGRU import MinGRUConfig
from minGRU_pytorch.minGRU import minGRU

class MinGRUWrapped(nn.Module):
    def __init__(self, min_gru_model):
        super().__init__()
        self.min_gru_model = min_gru_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, *args, **kwargs):
        args = [arg.to(self.device) if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return self.min_gru_model(*args, **kwargs)

    def to(self, device):
        self.device = device
        self.min_gru_model.to(device)
        return self

class MinGRUPreTrainedModel(PreTrainedModel):
    config_class = MinGRUConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        for name, param in module.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter {name}. Replacing with a safe number.")
                param.data = torch.nan_to_num(param.data, nan=1e-6)

class MinGRUForSequenceClassification(PreTrainedModel):
    config_class = MinGRUConfig
    base_model_prefix = "model"

    def __init__(self, config: MinGRUConfig):
        super().__init__(config)

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        raw_min_gru = minGRU(
            dim=config.d_model,
            expansion_factor=config.ff_mult
        )
        self.model = MinGRUWrapped(raw_min_gru)

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.d_model, config.num_labels)
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs
    ):
        embeddings = self.embedding(input_ids)

        logits = self.model(embeddings)

        pooled_output = logits.mean(dim=1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )

    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        for name, param in model.named_parameters():
            if name in ['embedding.weight', 'model.min_gru_model.to_hidden_and_gate.weight', 'model.min_gru_model.to_out.weight']:
                if param is None or torch.isnan(param).any() or torch.isinf(param).any():
                    nn.init.xavier_normal_(param)  # Başlatma işlemi
                    print(f"Initialized parameter {name} manually.")
    
        return model
    """

    def save_pretrained(self, save_directory, safe_serialization: Optional[bool] = True, **kwargs):
        """
        Save the model and configuration to a directory.
    
        Args:
            save_directory (str): Directory to save the model.
            safe_serialization (bool, optional): Whether to use safe serialization. Defaults to True.
            kwargs: Additional arguments like max_shard_size (ignored in this implementation).
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        if safe_serialization:
            print("Saving with safe serialization.")
            
            state_dict = {}
    
            for name, param in self.model.min_gru_model.named_parameters():
                state_dict[f"model.{name}"] = param
    
            for name, param in self.classifier.named_parameters():
                state_dict[f"classifier.{name}"] = param
    
            state_dict['config'] = self.config.__dict__
            torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
            
            self.config.save_pretrained(save_directory)
        else:
            print("Saving without safe serialization.")
            super().save_pretrained(save_directory)
