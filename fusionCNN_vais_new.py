import torch
import torch.nn as nn
from model_cnn_vais_new import Net_227_new,Net_227_ir
import torch.nn.functional as F

class FushionCNN(nn.Module):
    def __init__(self, pretrained=False, modelList=None):
        super(FushionCNN,self).__init__()
        self.net227_ir=Net_227_ir()
        self.net227=Net_227_new()
        if pretrained:
            stateDictList=[torch.load(modelFile) for modelFile in modelList]
            self.net227_ir.load_state_dict(stateDictList[0])
            self.net227.load_state_dict(stateDictList[1])
        self.net227_ir.eval()
        self.net227.eval()

    def forward(self, x_stacked):
        x227_ir=x_stacked[0]
        x227 = x_stacked[1]
        result=torch.cat([torch.unsqueeze(self.net227_ir(x227_ir),0),torch.unsqueeze(self.net227(x227),0)],0)
        result=F.softmax(result,-1)
        return result

