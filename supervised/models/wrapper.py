import torch
from torch import nn



class WrapperModel(nn.Module):
    def __init__(self, classification ,segmentation):
        super().__init__()
        self.classification_model = classification
        self.segmentation_model = segmentation
        # self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, xb):
        self.classification_model.eval()
        self.segmentation_model.eval()
        
        out = self.classification_model(xb)
        # prob_dist = self.softmax(out).squeeze().tolist()
        _, preds = torch.max(out, dim=1)
        if preds.item() == 1:
            mask = self.segmentation_model(xb)
        else:
            mask = torch.zeros([1, 1, xb.shape[2], xb.shape[3]])
        return mask