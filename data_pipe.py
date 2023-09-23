from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from utils import train_transforms, val_transforms


class surg_data(Dataset):

    def __init__(self, im_path, mk_path,txt_names, transform = None,binary = True,segment=True):
        self.im_path = im_path
        self.mk_path = mk_path
        self.names = pd.read_csv(txt_names)
        self.transforms = transform
        self.binary = binary
        self.segment = segment

    def __len__(self):
        # self.names = self.names[~self.names['names'].str.startswith('aug')] #remove upsamples
        return len(self.names)

    def __getitem__(self, index):
        # print(self.names.iloc[index,3])
        img_path = os.path.join(self.im_path,self.names.iloc[index,3] + '.jpg')
        mask_path = os.path.join(self.mk_path,self.names.iloc[index,3] + '.png')
        img_name = self.names.iloc[index,3] + '.jpg'
        mask_name = self.names.iloc[index,3] + '.png'
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.segment:
            class_lb = self.names.iloc[index,5]
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask = mask/255.0 #aug images have 0 and 1, shouldnt be dividing by 255 when training as this is already applied when augment.py
        else:
            class_lb = self.names.iloc[index,5] - 1.0
            mask = 0
        # mask = mask[:,:,0]
        # mask[mask == 255.0] = 1.0

        if self.transforms:
        #   image = self.transforms(image)
        #   mask = self.transforms(mask)
            if self.segment:
                augmentations = self.transforms(image=image, mask=mask)
                image = augmentations["image"]
                mask = augmentations["mask"]
            else:
                augmentations = self.transforms(image=image)
                image = augmentations["image"]

        if self.binary is False:
            idx = [mask == 1]
            mask[idx] = class_lb
        
        # else:
        #     to_Ten = ToTensorV2()
        #     image_dic = to_Ten(image = image)
        #     mask_dic = to_Ten(image = mask)
        #     image = image_dic["image"].squeeze(0)
        #     mask = mask_dic["image"].squeeze(0)


        return (image,mask,img_name,mask_name,class_lb)
    
# # '/home/bsidiqi/cluster_docs.nosync/imgs_frms/frames_final'
# # '/home/bsidiqi/cluster_docs.nosync/imgs_frms/masks_bi_29'
# #  '/home/bsidiqi/cluster_docs.nosync/imgs_frms/train_names.csv'
# # '/home/bsidiqi/cluster_docs.nosync/imgs_frms/test_names.csv'

# '/home/bilal/cluster_docs.nosync/imgs_frms/frames_final_26j'
# '/home/bilal/cluster_docs.nosync/imgs_frms/masks_bi_29_26j'
#  '/home/bilal/cluster_docs.nosync/imgs_frms/train_names_26j.csv'
# '/home/bilal/cluster_docs.nosync/imgs_frms/test_names_26j.csv'
#  '/home/bilal/cluster_docs.nosync/imgs_frms/aug300plus_orig.csv'
# '/home/bilal/cluster_docs.nosync/imgs_frms/test_names.csv'
# '/home/bilal/cluster_docs.nosync/imgs_frms/test_names_4cls.csv'

def loaders(binary,segment,im_path,mk_path,tr_names,te_names):
    train_data = surg_data(im_path,mk_path,tr_names,transform = train_transforms,binary=binary,segment=segment)
    train_loader = DataLoader(train_data, batch_size = 16, shuffle = True)

    test_data = surg_data(im_path,mk_path,te_names,transform = val_transforms,binary=binary,segment=segment)
    test_loader = DataLoader(test_data, batch_size = 16, shuffle = False)

    return train_loader, test_loader

# FOR AUG
# '/home/bilal/cluster_docs.nosync/imgs_frms/frames_final'
# '/home/bilal/cluster_docs.nosync/imgs_frms/masks_bi_29'
# aug_names = '/home/bilal/cluster_docs.nosync/imgs_frms/data_aug_fold3.csv'


# def loaders(binary=False,segment=True):
#     aug_data = surg_data(im_path,mk_path,aug_names,transform = train_transforms,binary=binary,segment=segment)
#     aug_loader = DataLoader(aug_data, batch_size = 16, shuffle = False)

#     return aug_loader


# For bnd box 
# '/home/bilal/cluster_docs.nosync/imgs_frms/frames_final'
# '/home/bilal/cluster_docs.nosync/imgs_frms/masks_bi_29'
# aug_names = '/home/bilal/cluster_docs.nosync/imgs_frms/test_names_26j.csv'


# def loaders(binary=True,segment=True):
#     data = surg_data(im_path,mk_path,aug_names,transform = val_transforms,binary=binary,segment=segment)
#     loader = DataLoader(data, batch_size = 16, shuffle = False)

#     return loader


def paths():

    ps =     [['/home/bilal/cluster_docs.nosync/imgs_frms/frames_final_26j',
    '/home/bilal/cluster_docs.nosync/imgs_frms/masks_bi_29_26j',
     '/home/bilal/cluster_docs.nosync/csv_files/train_names_26j.csv',
    '/home/bilal/cluster_docs.nosync/csv_files/test_names_26j.csv'],

    ['/home/bilal/cluster_docs.nosync/imgs_frms/frames_final_fold1',
    '/home/bilal/cluster_docs.nosync/imgs_frms/masks_bi_29_fold1',
     '/home/bilal/cluster_docs.nosync/csv_files/train_names_fold1.csv',
    '/home/bilal/cluster_docs.nosync/csv_files/test_names_fold1.csv'],

    ['/home/bilal/cluster_docs.nosync/imgs_frms/frames_final_fold2',
    '/home/bilal/cluster_docs.nosync/imgs_frms/masks_bi_29_fold2',
     '/home/bilal/cluster_docs.nosync/csv_files/train_names_fold2.csv',
    '/home/bilal/cluster_docs.nosync/csv_files/test_names_fold2.csv'],

    ['/home/bilal/cluster_docs.nosync/imgs_frms/frames_final_fold3',
    '/home/bilal/cluster_docs.nosync/imgs_frms/masks_bi_29_fold3',
     '/home/bilal/cluster_docs.nosync/csv_files/train_names_fold3.csv',
    '/home/bilal/cluster_docs.nosync/csv_files/test_names_fold3.csv']]

    return ps