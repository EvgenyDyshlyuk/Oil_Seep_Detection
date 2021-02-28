import torch
import torch.nn as nn
import torch.nn.functional as F

# Good implementation weight, size_average not implemented, so removed:
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
# Good description
# https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5




class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, preds, targets):

        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = F.sigmoid(preds)       
        
        #flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
 
        bce = F.binary_cross_entropy(preds, targets, reduction='mean')
        
        return bce


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, preds, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = F.sigmoid(preds)       
        
        #flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()                            
        dice = (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)  
        
        return (1 - dice)

    
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight

    def forward(self, preds, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = F.sigmoid(preds)       
        
        #flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        intersection = (preds * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)  
        bce = F.binary_cross_entropy(preds, targets, reduction='mean')
        dice_bce = bce*self.bce_weight + (1-self.bce_weight)*dice_loss
        
        return dice_bce

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, preds, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = F.sigmoid(preds)       
        
        #flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (preds * targets).sum()
        total = (preds + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha = 0.5, beta = 0.5, gamma = 1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, preds, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        preds = torch.sigmoid(preds)      
        
        #flatten label and prediction tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (preds * targets).sum()    
        FP = ((1-targets) * preds).sum()
        FN = (targets * (1-preds)).sum()
        
        tversky_index = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        focal_tversky = (1 - tversky_index)**self.gamma
                       
        return focal_tversky