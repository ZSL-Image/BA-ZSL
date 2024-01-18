import torch

def make_optimizer(cfg, model):

    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    momentum = cfg.SOLVER.MOMENTUM

    params_to_update = []
    params_names = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            params_names.append(name)

    optimizer = torch.optim.SGD(params_to_update, lr=lr,
                weight_decay=weight_decay, momentum=momentum)

    return optimizer

# def lr_lambda(s):
#     # base_lr = 5e-4
#     if s == 1:
#         lr = 1
#     elif s <= 2:
#         lr = 0.2
#     # elif s <= 3:
#     #     lr = base_lr * 0.2
#     else:
#         lr = 0.2 * 0.2
#     return lr
def lr_lambda(epoch):  
    base_lr = 5e-4  
    if epoch < 5:  
        lr = base_lr  
    elif epoch < 10:  
        lr = base_lr * 0.1   
    else:  
        lr = base_lr * 0.1 * 0.1  
    return lr

def make_lr_scheduler(cfg, optimizer):
    # def lr_lambda(s):
    #     base_lr = 5e-4
    #     if s < 5:
    #         lr = base_lr
    #     elif s < 10:
    #         lr = base_lr * 0.1 
    #     # elif s <= 3:
    #     #     lr = base_lr * 0.2
    #     else:
    #         lr = base_lr * 0.1 * 0.1
    #     return lr
    # return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    step_size = cfg.SOLVER.STEPS
    gamma = cfg.SOLVER.GAMMA
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)