import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA_Config:
    def __init__(self, r, lora_alpha, lora_dropout, merge_weights, target_modules):
        ## Your code ##
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        self.target_modules = target_modules


class LoRALayer(nn.Module):
    def __init__(self, original_layer, config: LoRA_Config):
        ## Your code ##
        self.original_layer = original_layer
        input_dim = original_layer.weight.size(1)
        output_dim = original_layer.weight.size(0)

        lora_A_tensor = torch.empty(config.r, input_dim)
        torch.nn.init.kaiming_uniform(lora_A_tensor)
        self.lora_A = nn.Parameter(lora_A_tensor)
        self.lora_B = nn.Parameter(torch.zeros(output_dim, config.r))
        self.scaling = config.lora_alpha / config.r

        if config.lora_dropout > 0:
            self.dropout = nn.Dropout(p=config.lora_dropout)
        else:
            self.dropout = lambda x : x
        

    def forward(self, x):
        ## Your code ##
        A_dropout = self.dropout(self.lora_A)
        B_dropout = self.dropout(self.lora_B)
        W_prime = self.original_layer.weight + self.scaling*B_dropout@A_dropout
        return F.linear(x, W_prime, self.original_layer.bias)


    def __repr__(self):
        return f'{self.__class__.__name__}(\n  (original_layer): {self.original_layer},\n  (lora_A): Parameter of size {self.lora_A.size()},\n  (lora_B): Parameter of size {self.lora_B.size()}\n)'


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.parameters():
        all_param += param.numel()
        if param.requires_gread:
            trainable_params += param.numel()
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {100* trainable_params / all_param}"
    )
    return trainable_params, all_param
