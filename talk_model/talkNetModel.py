import torch
import torch.nn as nn


from talk_model.visualEncoder     import visualFrontend, visualTCN, visualConv1D

class talkNetModel(nn.Module):
    def __init__(self):
        super(talkNetModel, self).__init__()
        # Visual Temporal Encoder
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        # self.visualFrontend.load_state_dict(torch.load('visual_frontend.pt', map_location="cuda"))
        # for param in self.visualFrontend.parameters():
        #     param.requires_grad = False       
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

    def forward(self, x):
        B, T, W, H = x.shape  #(13,117,112,112)
        x = x.view(B*T, 1, 1, W, H)#(13*117=1521,1,1,112,112)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x) #(1521,1,512),(64*1,1,512)<-since batch=1
        x = x.view(B, T, 512)  #(13,117,512)    (1,64,512)  
        x = x.transpose(1,2) #(13,512,117)    
        x = self.visualTCN(x)#(13,512,117)    
        x = self.visualConv1D(x)#(13,128,117)  
        x = x.transpose(1,2)#(13,117,128)
        return x

    