
import torch
from mmseg.models.builder import LOSSES



@LOSSES.register_module()
class CropBalanceloss(torch.nn.Module):
    def __init__(self,num_classes,with_learnable_weight=False,balance_weight=1,
                 average=True,ignore_index=-1):
        super().__init__()
        self.criterion=torch.nn.CrossEntropyLoss(reduction='none',ignore_index=ignore_index)
        self.with_learnable_weight=with_learnable_weight
        self.num_classes=num_classes
        if with_learnable_weight:
            self.weight=torch.nn.Parameter(torch.rand(num_classes),requires_grad=True)
        self.balance_weight=balance_weight
        self.average=average
    def forward(self,pred,label):
        loss=self.criterion(pred,label)
        # print(torch.unique(label),loss.mean())
        if self.with_learnable_weight:
            loss_list=[]
            for i in range(self.num_classes):
                mask=label==i
                fmask=mask.float()
                item_loss=loss*fmask
                item_loss=torch.sum(item_loss)/(torch.sum(fmask)+1e-6)
                loss_list.append(item_loss)
            loss=torch.stack(loss_list)
            weight=torch.nn.functional.softmax(self.weight)
            loss=loss*weight
            if self.average:
                loss=torch.sum(loss)
        else:
            if self.average:
                mask=label>=0
                loss=torch.sum(loss*mask)/(torch.sum(mask)+1e-6)
        return loss
