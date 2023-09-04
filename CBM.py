import os
import json
import torch

import torch.nn as nn


class MLP(nn.Module):
      def __init__(input_dim,latent_dim):
          super(MLP,self).__init__()
          self.linear=nn.Linear(input_dim,latent_dim)
          self.activation=nn.ReLU()
      def forward(self,x):
          x=self.linear(x)
          x=self.activation(x)
          return x


class CBM(nn.Module):
      def __init__(self,n_concepts)





class SeqCBM(nn.Module):
    def __init__(self,model1,model2,use_relu=False,use_sigmoid=False):
        super(SeqCBM,self).__init__()
        self.first_model=model1
        self.sec_model=model2
        self.use_relu=use_relu
        self.use_sigmoid=use_sigmoid
     def forward_stage(self,stage1_out):
         if self.use_relu:
            pred_out=[nn.ReLU()(o) for o in stage1_out]
         elif self.use_sigmoid:
              pred_out=[nn.Sigmoid()(o) for o in stage1_out]
         else:
             pred_out=stage1_out
         stage2_inputs=pred_out
         stage2_inputs = torch.cat(stage2_inputs, dim=1)
        all_out = [self.sec_model(stage2_inputs)]
        all_out.extend(stage1_out)
        return all_out
    def forward(self,x):
        if self.first_model.training:
           outputs, aux_outputs = self.first_model(x)
           return self.forward_stage2(outputs), self.forward_stage2(aux_outputs)
        else:
            outputs = self.first_model(x)
            return self.forward_stage2(outputs) 
