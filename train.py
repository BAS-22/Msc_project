import torch
import torch.nn as nn
import torchvision
import numpy as np
import pandas as pd
from torchmetrics.classification import JaccardIndex
from torch.autograd import Variable
from data_pipe import loaders, paths
import torchvision.transforms.functional as TF
from utils import DiceLoss
from utils import iou_pytorch
from utils import pc_iou_pytorch
from utils import FocalLoss
from unet_arc import unet 
from segformer_arc import SegFormer
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from typing import List, Tuple
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sn
from torch.utils.tensorboard import SummaryWriter
import time
import segmentation_models_pytorch as smp
from transformers import SegformerForSemanticSegmentation,SegformerConfig,  ViTForImageClassification, SwinForImageClassification
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


"""# **TRAINING**"""

if __name__ == '__main__':
    binary = False
    segment = True
    paths = paths()
    f1tr_loader, f1te_loader = loaders(binary,segment,paths[0][0],paths[0][1],paths[0][2],paths[0][3])
    f2tr_loader, f2te_loader = loaders(binary,segment,paths[1][0],paths[1][1],paths[1][2],paths[1][3])
    f3tr_loader, f3te_loader = loaders(binary,segment,paths[2][0],paths[2][1],paths[2][2],paths[2][3])
    f4tr_loader, f4te_loader = loaders(binary,segment,paths[3][0],paths[3][1],paths[3][2],paths[3][3])
    tr_loaders = [f1tr_loader,f2tr_loader,f3tr_loader,f4tr_loader]
    te_loaders = [f1te_loader,f2te_loader,f3te_loader,f4te_loader]

    # for l,train_loader in enumerate(tr_loaders):
    #     dataiter = iter(train_loader)
    #     imgs, masks,im_path,mask_path,cl_label = next(dataiter)
    #     # imgs, masks = next(dataiter)
    #     if segment:
    #         print(imgs.size(),masks.size())
    #         print(torch.unique(masks[0]))
    #     else:
    #         print(imgs.size())
    #     # display_images(imgs[:12], masks[:12], num_cols=6)

    #     # TensorBoard
    #     writer = SummaryWriter(f'trained_models/binary/runs{l}')

        # GridPlots

        # img_grid = torchvision.utils.make_grid(imgs)
        # masks = masks.unsqueeze(1)
        # mask_rgb = torch.cat((masks, masks, masks), dim=1)
        # mask_grid = torchvision.utils.make_grid(mask_rgb)
        # writer.add_image('Frames',img_grid)
        # writer.add_image('Masks',mask_grid)
    
    def train(net,optimizer,criterion,epoch, binary,segment,train_loader,writer,model_name):
        total_steps = len(train_loader)
        e_loss = 0
        b_loss = 0
        e_acc = 0
        b_acc = 0
        total = 0
        for i, data in enumerate(train_loader, 0):

            inputs, masks,img_path,mask_path,cl_labels = data
            inputs = Variable(inputs.to(device)).to(torch.float32)
            if segment:
                labels = Variable(masks.to(device)).to(torch.float32)
            else:
                labels = cl_labels.type(torch.LongTensor)
                labels = Variable(labels.to(device))
                # labels = Variable(cl_labels.to(device)).to(torch.LongTensor)

            # print(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            if model_name == 'segformer':
                outputs = outputs.logits
            elif model_name == 'deeplab' or model_name == 'cdeeplab' :
                outputs = outputs['out']
            
            if segment:
                if binary:
                    outputs = torch.sigmoid(outputs)
                    outputs = TF.resize(outputs,(288,512))
                    outputs = outputs.squeeze(1)

                else:
                    outputs = TF.resize(outputs,(288,512))

            loss = criterion(outputs, labels)
            if segment:
                acc = iou_pytorch(outputs,labels,binary).item()
            else:
                outputs = F.softmax(outputs,1)
                predicted = torch.argmax(outputs, 1)
                acc = ( (predicted == labels).sum().item() / labels.size(0) )

            e_acc += acc
            b_acc += acc
            e_loss += loss.item()
            b_loss += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()

            if i % 50 == 49:
                # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, b_loss / 50))
                writer.add_scalar('Training Loss',b_loss/50,global_step = total_steps*epoch + i)
                writer.add_scalar('Training evaluation',100 * b_acc/50,global_step = total_steps*epoch + i)
                b_loss = 0.0
                b_acc = 0.0

        epoch_loss = e_loss / len(train_loader)
        epoch_acc = 100 * e_acc / len(train_loader)


        # if epoch+1 == 2:
        #     writer.add_hparams({'lr': r},{'loss':epoch_loss,'acc': epoch_iou})

        # step += 1

        return print(f'Dice loss for epoch {epoch+1} is {epoch_loss} \n mIoU for epoch {epoch+1} is {epoch_acc}')

    def test(net,criterion,binary,segment,test_loader,model_name):
        N = len(test_loader)
        e_loss = 0
        e_acc = 0
        p_e_acc = torch.zeros(5, dtype=torch.float32)
        count = torch.zeros(5)
        t_aug = 0
        t_naug = 0
        t_count = 0
        net.eval()
        # start_time_aug = time.time()
        for i, data in enumerate(test_loader, 0):
            inputs, masks,img_path,mask_path,cl_labels = data
            inputs = Variable(inputs.to(device)).to(torch.float32)
            labels = Variable(masks.to(device)).to(torch.float32)
            # if segment:
            #     labels = Variable(masks.to(device)).to(torch.float32)
            # else:
            #     labels = cl_labels.type(torch.LongTensor)
            #     labels = Variable(labels.to(device))

            with torch.no_grad():
                # start_time_naug = time.time()
                outputs = net(inputs)
                if model_name == 'segformer':
                    outputs = outputs.logits
                elif model_name == 'deeplab' or model_name == 'cdeeplab' :
                    outputs = outputs['out']

                # outputs = TF.resize(outputs,(288,512))
                # outputs = F.softmax(outputs,1)
                # outputs = torch.argmax(outputs,1)
                # ..
                # end_time = time.time()
                # end_time_aug = time.time()
                # inference_time_aug = end_time - start_time_aug
                # inference_time_naug = end_time - start_time_naug
                # print("Inference time aug: {:.4f} seconds".format(inference_time_aug))
                # print("Inference time naug: {:.4f} seconds".format(inference_time_naug))
                # t_aug += inference_time_aug
                # t_naug += inference_time_naug
                # t_count += 1
                # outputs = outputs.logits

                if segment:
                    if binary:
                        outputs = torch.sigmoid(outputs)
                        outputs = TF.resize(outputs,(288,512))
                        outputs = outputs.squeeze(1)

                    else:
                        outputs = TF.resize(outputs,(288,512))
                        p_acc = pc_iou_pytorch(outputs,labels) #list, has nan where class not present in any labels
                
                loss = criterion(outputs,labels)
                if segment:
                    acc = iou_pytorch(outputs,labels,binary).item()
                    if not binary:
                        p_e_acc = torch.add(p_e_acc.nan_to_num(),p_acc.nan_to_num())
                        non_nan_idx = ~torch.isnan(p_acc)
                        count[non_nan_idx] += 1
                else:
                    outputs = F.softmax(outputs,1)
                    predicted = torch.argmax(outputs, 1)
                    acc = ( (predicted == labels).sum().item() / labels.size(0) )

                e_acc += acc
                e_loss += loss.item()
                # start_time_aug = time.time()

        t_loss = e_loss / len(test_loader)
        t_acc = e_acc / len(test_loader)
        if not binary:
            t_p_acc = p_e_acc / count
        # t_loss = 0
        # t_acc = 0
        # t_p_acc = 0
        # avg_augt = t_aug/t_count
        # avg_naugt = t_naug/t_count
            return t_loss, t_acc , t_p_acc
        else:
            return t_loss, t_acc

    
                                                        # Model
    def get_model(model_name,pretrained = True,binary=True):
        if binary:
            num_cls = 1
        else:
            num_cls = 5

        if pretrained:
            if model_name == 'unet':
                model = smp.Unet(
                    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes= num_cls,                      # model output channels (number of classes in your dataset)
                )
                for parm in model.encoder.parameters():
                    parm.requires_grad = False

            if model_name == 'deeplab':
                model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
                for parm in model.backbone.parameters():
                    parm.requires_grad = False
                model.classifier[4] = torch.nn.Conv2d(256, num_cls, kernel_size=(1, 1))

            if model_name == 'segformer':
                model_checkpoint = "nvidia/mit-b0"
                if binary:
                    id2label = {1: "instrument"}
                else:
                    id2label = {0:'background',1: "blunt dissector",2:'kerrisons',3:'pituitary rongeurs', 4:'cup forceps'}
                label2id = {label: id for id, label in id2label.items()}
                num_labels = len(id2label)
                model = SegformerForSemanticSegmentation.from_pretrained(
                    model_checkpoint,
                    num_labels=num_labels,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                )
        
        else:
            if model_name == 'unet':
                model = smp.Unet(
                    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                    classes=num_cls,                      # model output channels (number of classes in your dataset)
                )

            if model_name == 'deeplab':
                model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
                model.classifier[4] = torch.nn.Conv2d(256, num_cls, kernel_size=(1, 1))

            if model_name == 'cdeeplab':
                model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
                # s = print(model)
                # a=z
                aspp = model.classifier[0].convs
                og  = [(12,12),(24,24),(48,48)]
                # dil = [(2,2),(4,4),(8,8)]
                dil = [(14,14),(28,28),(56,56)]
                # dil = [(1,1),(1,1),(1,1)]
                aspp[1][0] = nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=dil[0], dilation=dil[0], bias=False)
                aspp[2][0] = nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding= dil[1], dilation=dil[1], bias=False)
                aspp[3][0] = nn.Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding= dil[2], dilation=dil[2], bias=False)
                # model.classifier[4] = nn.Sequential(nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                #                                     nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                #                                     nn.ReLU(),
                #                                     nn.Conv2d(256, num_cls, kernel_size=(1, 1)))
                model.classifier[4] = nn.Conv2d(256, num_cls, kernel_size=(1, 1))
                    
            if model_name == 'segformer':
                config = SegformerConfig()
                config.num_labels = num_cls
                model = SegformerForSemanticSegmentation(config=config)
        return model


    # model = get_model(model_name='deeplab',pretrained = False,binary=False)
    # print(model)
    # a=z
        # model_name = 'google/vit-base-patch16-224-in21k'
        # model_name = "microsoft/swin-tiny-patch4-window7-224"
        # id2label = {0: "blunt dissector",1:'kerrisons',2:'pituitary rongeurs', 3:'cup forceps'}
        # label2id = {label: id for id, label in id2label.items()}
        # num_labels = len(id2label)
        # model = model = SwinForImageClassification.from_pretrained(
        #     model_name,
        #     # image_size=(512, 512),
        #     num_labels=num_labels,
        #     id2label=id2label,
        #     label2id=label2id,
        #     ignore_mismatched_sizes=True,
        # )


                                            # TRAINING
    # alpha = [0.18603965996332458,7.3364236855341725,3.9866722954481735,10.233920897444726,11.512123783951019,18.839175976867946]
    # for j, alpha in enumerate(alphas):
    #     for gamma in gammas:
    #         writer = SummaryWriter(f'trained_models/runs/AlphaValue {j} GammaValue {gamma}')

    # for l,t_loader in enumerate(tr_loaders):
    # t_loader = tr_loaders[3]
    # l = 3
    # model = get_model(model_name='cdeeplab',pretrained=False,binary=False)
    # net = model
    # net.to(device)
    # # loss = DiceLoss()
    # # loss = nn.CrossEntropyLoss()
    # loss = FocalLoss()
    # # loss = nn.CrossEntropyLoss(weight = torch.tensor(
    # #    [0.8518518518518519,0.8562259306803595,1.0845528455284552,1.3584521384928716]).to(device).to(torch.float32))
    # # loss = nn.CrossEntropyLoss()
    # # lr = 2e-4
    # lr = 0.00006
    # optim = torch.optim.Adam(params = net.parameters(),lr = lr, weight_decay=1e-8)
    # writer = SummaryWriter(f'trained_models/multi/cdeeplab/runs3{l}')

    # for i in range(50):
    #     train(net,optim,loss,i,binary=False,segment=True,train_loader=t_loader,writer=writer,model_name='cdeeplab')
    #     if i % 10 == 9:
    #         # torch.save(net.state_dict(),f'/home/bsidiqi/cluster_docs.nosync/unet_10j/unet_10j_e{i+1}.pt')
    #         torch.save(net.state_dict(),f'/home/bilal/cluster_docs.nosync/trained_models/multi/cdeeplab/cdeeplab_db_f{l}/c3deeplab_19a_e{i+1}.pt')
    #         # torch.save(net.state_dict(),f'/home/bilal/cluster_docs.nosync/trained_models/tuning/{j}{gamma}.pt')
    
    # print(f'fold {l}:done')


    #                                                     # TESTING

    # for l,te_loader in enumerate(te_loaders[0:1]):
    # te_loader = te_loaders[3]
    # l = 3
    # model = get_model(model_name='unet',pretrained=False,binary=False)
    # print(model)
    # a=z
    # net = model
    # # net.load_state_dict(torch.load('/home/bilal/cluster_docs.nosync/trained_models/binary/seg/seg_b0_f0/pseg_19a_e20.pt'))
    # net.load_state_dict(torch.load(f'/home/bilal/cluster_docs.nosync/trained_models/multi/cdeeplab/cdeeplab_db_f{l}/c2deeplab_19a_e40.pt'))
    # net.to(device)
    # # loss = DiceLoss()
    # loss = FocalLoss()
    # # loss = nn.CrossEntropyLoss(weight = torch.tensor(
    # #    [0.6630513376717281, 0.4633653360282971, 4.265116279069767, 10.076923076923077]).to(device).to(torch.float32))
    # # loss = nn.CrossEntropyLoss()
    # # loss, miou, per_class_miou,avg_augt,avg_naugt = test(net,loss,binary=True,segment=True)
    # loss, miou, per_class_miou= test(net,loss,binary=False,segment=True,test_loader=te_loader,model_name='cdeeplab')
    # # print(loss, miou, per_class_miou,f'aug:{avg_augt}',f'naug:{avg_naugt}')
    # print(loss, miou,per_class_miou)

                                                # PLOTTING

    def display_te_images(images, ground_truth, pred_unet,pred_deeplab,pred_seg, figsize: Tuple[int, int]=(15, 10), num_cols:int=5,binary = False):
        
        print(images.shape)
        print(ground_truth.shape)
        print(pred_unet.shape)

        images = (images).numpy().transpose((0, 2, 3, 1))
        images = [Image.fromarray((images[i]).astype('uint8')) for i in range(len(images))]
        ground_truth = (ground_truth).numpy()
        ground_truth = [Image.fromarray((ground_truth[i]).astype('uint8'), 'L') for i in range(len(ground_truth))]

        
        if binary:
            pred_unet = (pred_unet * 100).numpy().transpose((0, 2, 3, 1))
            pred_unet = [Image.fromarray((pred_unet[i].squeeze()).astype('uint8'), 'L') for i in range(len(pred_unet))]
            pred_deeplab = (pred_deeplab* 100).numpy().transpose((0, 2, 3, 1))
            pred_deeplab = [Image.fromarray((pred_deeplab[i].squeeze()).astype('uint8'), 'L') for i in range(len(pred_deeplab))]
            pred_seg = (pred_seg * 100).numpy().transpose((0, 2, 3, 1))
            pred_seg = [Image.fromarray((pred_seg[i].squeeze()).astype('uint8'), 'L') for i in range(len(pred_seg))]
        else:
            pred_unet = (pred_unet).numpy()
            pred_unet = [Image.fromarray((pred_unet[i]).astype('uint8'), 'L') for i in range(len(pred_unet))]
            pred_deeplab = (pred_deeplab).numpy()
            pred_deeplab = [Image.fromarray((pred_deeplab[i]).astype('uint8'), 'L') for i in range(len(pred_deeplab))]
            pred_seg = (pred_seg).numpy()
            pred_seg = [Image.fromarray((pred_seg[i]).astype('uint8'), 'L') for i in range(len(pred_seg))]
            

    
        num_images = len(images) + len(ground_truth) + len(pred_unet) + len(pred_deeplab) + len(pred_seg)

        # Create a grid of subplots for the ground_truth
        num_cols = num_cols
        num_rows = int(np.ceil(num_images / num_cols))
        cols = ['GT_imgs','GT_masks','Pred_U-Net','Pred_DeepLabv3','Pred_SegFormer']
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
        for ax, col in zip(axs[0], cols):
            ax.set_title(col,fontweight="bold")

        axs = axs.flatten()

        img_idx = 0
        class_colors = ['purple', 'green', 'blue','orange','yellow']
        cmap = matplotlib.colors.ListedColormap(class_colors)
        for i in range(0, num_images, 5):
            axs[i].imshow(images[img_idx])
            axs[i+1].imshow(ground_truth[img_idx],cmap = cmap, vmin=0, vmax=len(class_colors)-1)
            # axs[i+1].imshow(ground_truth[img_idx])
            a = axs[i+2].imshow(pred_unet[img_idx],cmap = cmap, vmin=0, vmax=len(class_colors)-1)
            # axs[i+2].imshow(pred_unet[img_idx],cmap = cmap, vmin=0, vmax=len(class_colors)-1)
            axs[i+3].imshow(pred_deeplab[img_idx],cmap = cmap, vmin=0, vmax=len(class_colors)-1)
            axs[i+4].imshow(pred_seg[img_idx],cmap = cmap, vmin=0, vmax=len(class_colors)-1)
            fig.colorbar(a)

            # plt.xlabel('GT_imgs')
            # plt.xlabel('GT_masks')
            # plt.xlabel('Pred_unet')
            # plt.xlabel('Pred_deeplab')
            # plt.xlabel('Pred_segformer')

            axs[i].axis('off')
            axs[i+1].axis('off')
            axs[i+2].axis('off')
            axs[i+3].axis('off')
            axs[i+4].axis('off')
            img_idx += 1

            # if i == num_images-1:

        plt.savefig('/home/bilal/cluster_docs.nosync/trained_models/multi/test.png')
    
    def plot(model_names,binary):
        dataiter = iter(te_loaders[1])
        te_imgs, te_masks ,im_path,mask_path,cl_labels= next(dataiter)
        # print(te_imgs.size(),te_masks.size())
        inputs = Variable(te_imgs.to(device)).to(torch.float32)
        labels = Variable(te_masks.to(device)).to(torch.float32)
        preds = [] 
        for m,model_name in enumerate(model_names):
            if model_name == 'segformer':
                pt = True 
            else:
                pt = False

            with torch.no_grad():
                # print(model_name)
                model = get_model(model_name=model_name,pretrained=pt,binary=binary)
                net = model
                # net.load_state_dict(torch.load('/home/bilal/cluster_docs.nosync/trained_models/binary/seg/seg_b0_f0/pseg_19a_e20.pt'))
                if model_name == 'segformer':
                    net.load_state_dict(torch.load(f'/home/bilal/cluster_docs.nosync/trained_models/multi/seg/seg_f1/seg_26j_e50.pt'))
                else:
                    net.load_state_dict(torch.load(f'/home/bilal/cluster_docs.nosync/trained_models/multi/{model_name}/{model_name}_f1/{model_name}_19a_e50.pt'))
                net.to(device)
                outputs = net(inputs)
                if model_name == 'segformer':
                    outputs = outputs.logits
                elif model_name == 'deeplab':
                    outputs = outputs['out']
                # print(outputs.logits).size()
            if binary:
                outputs = (outputs >0.5).float()
                outputs = outputs.cpu()
            else:
                outputs = TF.resize(outputs,(288,512))
                outputs = F.softmax(outputs,1)
                outputs = torch.argmax(outputs,1)
                outputs = outputs.cpu()
            
            preds.append(outputs)
            
        inputs = inputs.cpu()
        print(len(preds))
        display_te_images(inputs[:4],te_masks[:4], pred_unet=preds[0][:4],pred_deeplab=preds[1][:4],pred_seg=preds[2][:4], num_cols=5,binary = binary)

    model_names = ['unet','deeplab','segformer']
    plot(model_names=model_names,binary=False)


    # def display_te_images(images, masks, figsize: Tuple[int, int]=(15, 10), num_cols:int=6,binary = False):
    #     matplotlib.use('Agg')
    #     # images = (images / 2 + 0.5).numpy().transpose((0, 1, 2, 3))
    #     print(images.shape)
    #     print(masks.shape)
    #     images = (images).numpy()
    #     # print(np.unique(masks[0]))
    #     images = [Image.fromarray((images[i]).astype('uint8'), 'L') for i in range(len(images))]
    #     # masks = np.where(masks==2,2,0)
    #     # masks = (masks * 100).transpose((0, 2, 3, 1))
    #     # masks = (masks * 100).numpy().transpose((0, 1, 2, 3))

        
    #     if binary:
    #         masks = (masks * 100).numpy().transpose((0, 2, 3, 1))
    #         masks = [Image.fromarray((masks[i].squeeze()).astype('uint8'), 'L') for i in range(len(masks))]
    #     else:
    #         masks = (masks).numpy()
    #         # print(np.unique(masks[0]),np.unique(masks[1]),np.unique(masks[1]))
    #         masks = [Image.fromarray((masks[i]).astype('uint8'), 'L') for i in range(len(masks))]

    
    #     num_images = len(images) + len(masks)

    #     # Create a grid of subplots for the images
    #     num_cols = num_cols
    #     num_rows = int(np.ceil(num_images / num_cols))

    #     fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    #     axs = axs.flatten()

    #     img_idx = 0
    #     class_colors = ['purple', 'green', 'blue','orange','black','yellow']
    #     cmap = matplotlib.colors.ListedColormap(class_colors)
    #     for i in range(0, num_images, 2):
    #         axs[i].imshow(images[img_idx],cmap = cmap, vmin=0, vmax=len(class_colors)-1)
    #         a = axs[i+1].imshow(masks[img_idx],cmap = cmap, vmin=0, vmax=len(class_colors)-1)
    #         fig.colorbar(a)

    #         axs[i].axis('off')
    #         axs[i+1].axis('off')

    #         img_idx += 1

        # plt.savefig('/home/bilal/cluster_docs.nosync/trained_models/pu_mseg_b0_18j/test.png')

    # display_te_images(inputs[:12],te_masks[:12], outputs[:12], num_cols=6,binary = False)

                                                    # Confusion Matrix
    classes = ['Blunt Dissector', 'Kerrisons',
            'Pituitary Rongeurs', 'Cup Forceps']
    def ConfusionM(model_name,segment=True,te_loaders = te_loaders,classes=classes):
        cms = np.zeros((4,5,5))
        if model_name == 'segformer':
            pt = True 
        else:
            pt = False

        for l,te_loader in enumerate(te_loaders):
            model = get_model(model_name=model_name,pretrained=pt,binary=False)
            net = model
            if model_name == 'segformer':
                net.load_state_dict(torch.load(f'/home/bilal/cluster_docs.nosync/trained_models/multi/seg/seg_f{l}/seg_26j_e50.pt'))
            else:
                net.load_state_dict(torch.load(f'/home/bilal/cluster_docs.nosync/trained_models/multi/{model_name}/{model_name}_f{l}/{model_name}_19a_e30.pt'))
        
            net.to(device)
            y_pred = []
            y_true = []
            net.eval()
            for i, data in enumerate(te_loader, 0):
                inputs, masks,img_path,mask_path,cl_labels = data
                inputs = Variable(inputs.to(device)).to(torch.float32)
                # if segment:
                #     labels = Variable(masks.to(device)).to(torch.float32)
                # else:
                labels = cl_labels.type(torch.LongTensor)
                labels = Variable(labels.to(device))

                with torch.no_grad():
                    outputs = net(inputs)
                    if model_name == 'segformer':
                        outputs = outputs.logits
                    elif model_name == 'deeplab':
                        outputs = outputs['out']

                    if segment:
                        outputs = TF.resize(outputs,(288,512))
                    outputs = F.softmax(outputs,1)
                    outputs = torch.argmax(outputs,1)
                    
                if segment:
                    targets = torch.zeros(inputs.size(0))
                    for j in range(0,inputs.size(0)):
                        vl,cn = torch.unique(outputs[j], return_counts=True)
                        sorted, indices = torch.sort(cn,descending=True)
                        if indices.size(0) > 1:
                            if indices[1]!=0:
                                targets[j] = vl[indices[1]]
                            else:
                                targets[j] = vl[indices[0]]
                        else:
                            targets[j] = vl[indices[0]]


                labels = labels.cpu().numpy()
                if segment:
                    outputs = targets.cpu().numpy()
                else:
                    outputs = outputs.cpu().numpy()
                    
                # labels[targets==0] = 0
                y_pred.extend(outputs)
                y_true.extend(labels)
                # print(outputs)


            # classes = ['background','freer elevator', 'kerrisons',
            # 'pituitary rongeurs', 'spatula dissector', 'cup forceps']
            cf_matrix = confusion_matrix(y_true, y_pred)
            # print(cf_matrix)
            cms[l] = cf_matrix
            cf_matrix = cf_matrix[1:,1:]
            df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                            columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(f'/home/bilal/cluster_docs.nosync/trained_models/multi/{model_name}_f{l}_cm.png')

        return cms  
      
    fold_wts = np.array([[[0.25], [0.294027565], [0.237085582], [0.092827004], [0.241666667]],
                            [[0.25], [0.243491577], [0.31919815], [0.35021097], [0.283333333]],
                            [[0.25], [0.28330781],	[0.212798766], [0.202531646], [0.366666667]],
                            [[0.25], [0.179173047], [0.230917502], [0.35443038], [0.108333333]]])
    def avgCM(weights,model_name):
        CM = ConfusionM(model_name=model_name)
        print(CM.shape,weights.shape)
        wtd_CM = np.zeros((4,5,5))
        for i in range(4):
            wtd_CM[i] = weights[i]*CM[i].T

        print(wtd_CM.shape)

        avg_cm = np.sum(wtd_CM,axis = 0).T
        #std
        devis = np.zeros((4,5,5))
        for m in range(CM.shape[0]):
            devis[m] = (CM[m] - avg_cm)**2

        wtd_std = np.zeros((4,5,5))
        for i in range(4):
            wtd_std[i] = weights[i]*devis[i].T

        avg_std = np.sqrt(np.sum(wtd_std,axis = 0)).T
        
        print(avg_cm)
        print(avg_std)
        avg_cm = avg_cm[1:,1:]
        avg_std = avg_std[1:,1:]

        df_cm = pd.DataFrame(np.round(avg_cm / np.sum(avg_cm, axis=1)[:, None],decimals =2) , index = [i for i in classes],
                        columns = [i for i in classes])
        df_std = pd.DataFrame(np.round(100*(avg_std / avg_cm),decimals = 2) , index = [i for i in classes],
                        columns = [i for i in classes])
        df = pd.concat([df_cm[col].astype(str) + '±'+ df_std[col].astype(str) + '%' for col in df_cm], axis="columns")
        
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=df,fmt="",annot_kws={"size": 16})
        plt.savefig(f'/home/bilal/cluster_docs.nosync/trained_models/multi/{model_name}_avgcm.png')

    model_names = ['unet','deeplab','segformer']
    # model_names = ['unet','deeplab']
    for model_name in model_names:
        avgCM(fold_wts,model_name = model_name)
