def freeze_model_filtered(model, unfrozen_names=[], unfrozen_type=None):
    for param_name, param in model.named_parameters():
        if unfrozen_type is None: # freeze all param but unfrozen_names
            if all([(name not in param_name) for name in unfrozen_names]):
                param.requires_grad = False
        else: # keep only a type of params unfrozen within unfrozen_names, all others are frozen  
            if all([(name not in param_name) and (unfrozen_type not in param_name) for name in unfrozen_names]):
                param.requires_grad = False