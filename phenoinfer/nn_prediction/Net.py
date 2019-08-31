import torch.nn as nn
import torch.nn.functional as F
import torch



class Net1(nn.Module):


    def __init__(self):
        super(Net1,self).__init__()
        self.f1=nn.Linear(200,100)
        self.f2=nn.Linear(100,50)
        self.f3=nn.Linear(50,1)
        self.dropout=nn.Dropout(0.2)


    def forward(self, x):
        # first layer
        x=self.f1(x)
        x=F.selu(x)
        x=self.dropout(x)

        # second layer

        x=self.f2(x)
        x=F.selu(x)
        x=self.dropout(x)

        # third layer

        x=self.f3(x)
        x=torch.sigmoid(x)

        return x
