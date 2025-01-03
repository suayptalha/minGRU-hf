import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
from typing import Optional
from .configuration_minGRULM import MinGRULMConfig
from minGRU_pytorch.minGRULM import minGRULM


class MinGRULMWrapped(nn.Module):
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



class MinGRULMPreTrainedModel(PreTrainedModel):
    config_class = MinGRULMConfig
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

            
class MinGRULMForCausalLM(PreTrainedModel):
    config_class = MinGRULMConfig
    base_model_prefix = "model"

    def __init__(self, config: MinGRULMConfig):
        super().__init__(config)

        raw_min_gru = minGRULM(
            num_tokens=config.vocab_size,
            dim=config.d_model,
            depth=config.n_layer,
            ff_mult=config.ff_mult,
            min_gru_expansion=config.min_gru_expansion,
            enable_conv=config.enable_conv,
        )
        self.model = MinGRULMWrapped(raw_min_gru)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.post_init()

    def post_init(self):
        super().post_init()
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.min_gru_model.token_emb.weight

    def get_input_embeddings(self):
        return self.model.min_gru_model.token_emb

    def set_input_embeddings(self, value):
        self.model.min_gru_model.token_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask", None)}

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None, return_dict: Optional[bool] = True, **kwargs):
        logits = self.model(input_ids)

        if torch.isnan(logits).any():
            print("NaN detected in logits! Replacing with a safe number.")
            logits = torch.nan_to_num(logits, nan=1e-6)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

            if torch.isnan(loss).any():
                print("NaN detected in loss! Replacing with a safe number.")
                loss = torch.nan_to_num(loss, nan=1e-6)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Load model from a pretrained checkpoint.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def save_pretrained(self, save_directory, safe_serialization: Optional[bool] = True):
        """
        Save the model and configuration to a directory.
    
        Args:
            save_directory (str): Directory to save the model.
            safe_serialization (bool, optional): Whether to use safe serialization. Defaults to True.
        """
        os.makedirs(save_directory, exist_ok=True)
        
        if safe_serialization:
            print("Saving with safe serialization.")
            
            state_dict = {}

            for name, param in self.model.min_gru_model.named_parameters():
                state_dict[f"model.{name}"] = param
        
            for name, param in self.lm_head.named_parameters():
                state_dict[f"lm_head.{name}"] = param

            state_dict['config'] = self.config.__dict__
            torch.save(state_dict, os.path.join(save_directory, "pytorch_model.bin"))
            
            self.config.save_pretrained(save_directory)
        else:
            print("Saving without safe serialization.")
            super().save_pretrained(save_directory)
