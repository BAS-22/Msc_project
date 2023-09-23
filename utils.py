import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Iterable
import torchvision.transforms.functional as TF
from torchmetrics.classification import JaccardIndex
from torch.autograd import Variable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib
import cv2
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

IMAGE_HEIGHT = 448
IMAGE_WIDTH = 800
# IMAGE_HEIGHT = 512
# IMAGE_WIDTH = 896

# FOR AUG
# train_transforms = A.Compose(
#     [
#         # A.RandomCrop(540, 960, always_apply=False, p = 0.2),
#         # A.Resize(height= IMAGE_HEIGHT, width= IMAGE_WIDTH),
#         # A.CenterCrop(288, 512),
#         # A.Affine(scale = 1.1,keep_ratio=True),
#         A.Rotate(limit=90, p=0.5,border_mode = cv2.BORDER_CONSTANT),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
#         # A.Normalize(
#         #     mean=[45.4, 45.9, 66.4],
#         #     std=[65.2, 63.7, 82.7],
#         #     # max_pixel_value=255.0,
#         # ),
#         ToTensorV2(),
#     ],
# )


train_transforms = A.Compose(
    [
        # A.RandomCrop(540, 960, always_apply=False, p = 0.2),
        A.Resize(height= IMAGE_HEIGHT, width= IMAGE_WIDTH),
        # A.Resize(height= 137, width= 224),
        A.CenterCrop(288, 512),
        # A.CenterCrop(512, 512),
        # A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        # A.Affine(scale = 1.1,keep_ratio=True),
        A.Rotate(limit=35, p=0.5,border_mode = cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        # A.Normalize(
        #     mean=[45.4, 45.9, 66.4],
        #     std=[65.2, 63.7, 82.7],
        #     # max_pixel_value=255.0,
        # ),
        ToTensorV2(),
    ],
)

val_transforms = A.Compose(
    [   

        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # A.CenterCrop(288, 512),
        A.Resize(height=288, width=512),
        # A.Resize(height= 137, width= 224),
        # A.PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=[0, 0, 0]),
        # A.Normalize(
        #     mean=[45.4, 45.9, 66.4],
        #     std=[65.2, 63.7, 82.7],
        #     max_pixel_value=255.0,
        # ),
        ToTensorV2(),
    ],
)

jaccard = JaccardIndex(task="multiclass", num_classes=2)
jaccard.to(device)

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor,binary = True):
    # input_shape: (N,6,H,W)
    # labels_shape: (N,H,W)
    SMOOTH = 1e-6

    if binary:
      outputs = torch.where(outputs>0.5, 1,0)
    else:
      outputs = F.softmax(outputs,1)
      outputs = torch.argmax(outputs,1)

    intersection = (outputs.int() & labels.int()).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs.int() | labels.int()).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return torch.mean(iou)

class DiceLoss(nn.Module):
  def __init__(self, weight=None, size_average=True):
      super(DiceLoss, self).__init__()

  def forward(self, preds, targets, smooth=1):

    # outputs = torch.sigmoid(outputs)
    # outputs = TF.resize(outputs,(288,512))
    # outputs = (outputs > 0.5).float()
    # print(torch.unique(preds[0]))
    #flatten label and prediction tensors
    # preds = preds.view(-1)
    # targets = targets.view(-1)
    intersection = (preds * targets).sum()
    dice = (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)

    return torch.mean(1 - dice)
  

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        target = target.type(torch.int64)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
  


def display_te_images(images, masks, figsize: Tuple[int, int]=(15, 10), num_cols:int=6):
    matplotlib.use('Agg')
    # images = (images / 2 + 0.5).numpy().transpose((0, 1, 2, 3))
    print(images.size())
    images = (images * 100).numpy()
    # print(np.unique(masks[0]))

    images = [Image.fromarray((images[i].squeeze()).astype('uint8'), 'L') for i in range(len(images))]
    # masks = np.where(masks==2,2,0)
    # masks = (masks * 100).transpose((0, 2, 3, 1))
    # masks = (masks * 100).numpy().transpose((0, 1, 2, 3))
    masks = (masks * 100).numpy().transpose((0, 2, 3, 1))

    masks = [Image.fromarray((masks[i].squeeze()).astype('uint8'), 'L') for i in range(len(masks))]

    num_images = len(images) + len(masks)

    # Create a grid of subplots for the images
    num_cols = num_cols
    num_rows = int(np.ceil(num_images / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()

    img_idx = 0
    for i in range(0, num_images, 2):
        axs[i].imshow(images[img_idx])
        plt.xlabel('GT')
        axs[i+1].imshow(masks[img_idx])
        plt.xlabel('Pred')
        axs[i].axis('off')
        axs[i+1].axis('off')

        img_idx += 1


def pc_iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # input_shape: (N,6,H,W)
    # labels_shape: (N,H,W)
    SMOOTH = 1e-6
    num_classes = 5


    outputs = F.softmax(outputs,1)
    outputs = torch.argmax(outputs,1)

    iou_list = torch.zeros(num_classes, dtype=torch.float32)
    for class_id in range(num_classes):
        predicted_class = (outputs == class_id)
        target_class = (labels == class_id)
        intersection = (predicted_class.int() & target_class.int()).float().sum((1, 2))
        union = (predicted_class.int() | target_class.int()).float().sum((1, 2))
        iou = (intersection + SMOOTH) / (union + SMOOTH)  # iou is a list with miou of class_id for each example 
        absent = target_class.float().sum((1, 2)) == 0 # gives list of true and false 
        iou[absent] = float('nan')
        iou_list[class_id] = torch.nanmean(iou)

    # intersection = (outputs.int() & labels.int()).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    # union = (outputs.int() | labels.int()).float().sum((1, 2))         # Will be zzero if both are 0

    # iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds

    return iou_list