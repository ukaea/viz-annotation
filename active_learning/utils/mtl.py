import torch
from torch import nn

class MultiTaskLossWrapper(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        
        self.loss_cfg = loss_cfg

        self.log_vars = torch.nn.ParameterDict()
        self.learnable_tasks = []
        
        for loss_name, loss_dict in loss_cfg.items():
            if loss_dict['learn_weight']:
                self.log_vars[loss_name] = torch.nn.Parameter(torch.tensor(loss_dict['weight']), requires_grad=True)
                self.learnable_tasks.append(loss_name)

    def get_weight(self, task_name):
        if task_name in self.learnable_tasks:
            # return self.log_vars[task_name].detach()
            return 0.5/(self.log_vars[task_name]**2).detach()
        else:
            return torch.tensor(getattr(self.loss_cfg, task_name)['weight'])

    def forward(self, task_loss, task_name):

        if task_name in self.learnable_tasks:
            # precision = torch.exp(-self.log_vars[task_name])
            # loss = precision * task_loss + self.log_vars[task_name]
            task_loss = 0.5/(self.log_vars[task_name]**2 + 1e-9) * task_loss + torch.log(1 + self.log_vars[task_name] **2) 
        else:
            task_loss = task_loss * getattr(self.loss_cfg, task_name)['weight']
        
        return task_loss