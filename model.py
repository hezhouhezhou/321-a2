import torch.nn as nn
import torch
import torch.nn.functional as F
# define a linear classification model, divide each entry into two parts
# first part being the first 12 dimensions and second being the last 4 dimensions

class DryBeanModel(nn.Module):
    def __init__(self):
        super(DryBeanModel, self).__init__()
        self.ln = nn.Linear(16,7)
        self.soft = nn.Softmax(dim=1)

    def forward(self,inputs):
        return self.ln(inputs)
    
        
