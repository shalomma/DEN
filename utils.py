import numpy as np
import matplotlib.pyplot as plt


def params_to_update(model):
    print("Params to learn:")
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
            
    return params_to_update


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
