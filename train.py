import sys
import os
import argparse
import time
import numpy as np
import glob
import csv
from tree_util import create_tree_from_textfile, add_channels, add_levels, find_depth, getTreeList, update_channels

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.io import imread
from skimage.transform import resize

from Data import dataloaders
from Models import models
from Metrics import performance_metrics
from Metrics import losses
import math

# WARNING: THIS SUPPRESSES WARNINGS, REMOVE IF YOU WANT TO SEE WARNINGS
import warnings
warnings.filterwarnings("ignore")



def get_metrics(output, target, accuracy, IoU, dice, precision, recall, superIOU, superACC, superPERF, superPREC, superRECA, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics, args, get_loss, lossFuncts, t_root, loss):
        # level loss
        if get_loss:
            # convert output and target 
            if args.hierarchical_loss == True:
                loss1 = lossFuncts[0](output, target, t_root)[1]
                loss2 = lossFuncts[1](output, target, t_root)[1]
                if not loss:
                    for levels in range(len(loss1)):
                        loss.append(loss1[levels].item() + loss2[levels].item())
                else:
                    for levels in range(len(loss1)):
                        loss[levels] = (loss[levels] + (loss1[levels].item() + loss2[levels].item())) / 2
            
        
        # calculate metrics for the batch for each class
        newIOU = Iou(output, target, device, args.num_classes)
        newACC = Accuracy(output, target, device, args.num_classes)
        newPERF = perf_measure(output, target, device, args.num_classes)
        newPREC = Precision(output, target, device, args.num_classes)
        newRECA = Recall(output, target, device, args.num_classes)

        # updates the total metrics for all classes
        accuracy.append(newACC[0].item())
        IoU.append(newIOU[0].item())
        dice.append(newPERF[0].item())
        precision.append(newPREC[0].item())
        recall.append(newRECA[0].item())

        for clss in range(len(newACC[1])):
            # add calculates the per class metrics 
            clssMetrics[clss]['accuracy'].append(newACC[1][clss].item())
            clssMetrics[clss]['iou'].append(newIOU[1][clss].item())
            clssMetrics[clss]['dice'].append(newPERF[1][clss].item())
            clssMetrics[clss]['precision'].append(newPREC[1][clss].item())
            clssMetrics[clss]['recall'].append(newRECA[1][clss].item())
            
        
        # calculates superclass metrics for class 1 and 2
        if args.calc_super != False and args.include_background != False:
            target = [torch.where(x == 2, 1, x) for x in target]
            output1 = output[:,:1,:,:]
            output2 = (output[:,1,:,:] + output[:,2,:,:]).unsqueeze(1)
            output = torch.cat((output1, output2), 1)
            superIOU.append(Iou(output, target, device, (args.num_classes-1))[1][1].item())
            superACC.append(Accuracy(output, target, device, (args.num_classes-1))[1][1].item())
            superPERF.append(perf_measure(output, target, device, (args.num_classes-1))[1][1].item())
            superPREC.append(Precision(output, target, device, (args.num_classes-1))[1][1].item())
            superRECA.append(Recall(output, target, device, (args.num_classes-1))[1][1].item())


        return(clssMetrics, accuracy, IoU, dice, precision, recall, superIOU, superACC, superPERF, superPREC, superRECA, loss)


def train_epoch(model, device, train_loader, optimizer, epoch, lossFuncts, args, t_root, Accuracy, Iou, perf_measure, Precision, Recall):
    loss, accuracy, IoU, dice, precision, recall = [], [], [], [], [], []
    levelLoss = []
    

    clssMetrics = []

    # sets up dictionary for class metrics
    for clss in range(args.num_classes):
        # creates a dictionary in classMetrics
        clssMetrics.append({'accuracy': [], 'iou': [], 'dice': [], 'precision': [], 'recall': []})  # creates a dictionary for each class



    t = time.time()
    model.train()
    loss_accumulator = []
    for batch_idx, (data, target) in enumerate(train_loader):
        superIOU, superACC, superPERF, superPREC, superRECA = [], [], [], [], []

        data, target = data.to(device), [i.to(device) for i in target]
        optimizer.zero_grad()
        output = model(data)


        clssMetrics, accuracy, IoU, dice, precision, recall, superIOU, superACC, superPERF, superPREC, superRECA, _ = get_metrics(output, target, accuracy, IoU, dice, precision, recall, superIOU, superACC, superPERF, superPREC, superRECA, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics, args, False, None, None, None)



        # NEEDED FOR LOSS WITH CROSS ENTROPY
        if args.hierarchical_loss == True:
            loss1 = lossFuncts[0](output, target, t_root)
            loss2 = lossFuncts[1](output, target, t_root)
        else:
            loss1 = lossFuncts[0](output, target)
            loss2 = lossFuncts[1](output, target)


        # loss per level metrics
        if args.hierarchical_loss == True:
            if not levelLoss:
                for levels in range(len(loss1[1])):
                    levelLoss.append(loss1[1][levels].item() + loss2[1][levels].item())
            else:
                for levels in range(len(loss1[1])):
                    levelLoss[levels] = (levelLoss[levels] + (loss1[1][levels].item() + loss2[1][levels].item())) / 2


        # calculates the loss
        if args.hierarchical_loss == True:
            loss = loss1[0] + loss2[0]
        else:
            loss = loss1 + loss2
        try:
            loss.requires_grad = True
        except:
            pass
        

        loss.backward() #.item()
        optimizer.step()
        loss_accumulator.append(loss.item())
        if batch_idx + 1 < len(train_loader):
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    loss.item(),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTrain Epoch: {} [{}/{} ({:.1f}%)]\tAverage loss: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    (batch_idx + 1) * len(data),
                    len(train_loader.dataset),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    np.mean(loss_accumulator),
                    time.time() - t,
                )
            )
        
    # means the contents of each key in each dictionary in clssMetrics
    for met in range(len(clssMetrics)):
        for key in clssMetrics[met]:
            clssMetrics[met][key] = np.mean(clssMetrics[met][key])
        

    return np.mean(loss_accumulator).item(), clssMetrics, np.mean(accuracy).item(), np.mean(IoU).item(), np.mean(dice).item(), np.mean(precision).item(), np.mean(recall).item(), levelLoss


@torch.no_grad()
def test(model, device, test_loader, epoch, Accuracy, Iou, perf_measure, Precision, Recall, args, save_loc, lossFuncts, t_root):
    print('TESTING')
    IoU2 = []
    dice2 = []
    precision2 = []
    recall2 = []
    accuracy2 = []
    superIOU, superACC, superPERF, superPREC, superRECA = [], [], [], [], []

    clssMetrics2 = []

    inital = True

    # sets up dictionary for class metrics
    for clss in range(args.num_classes):
        # creates a dictionary in classMetrics
        clssMetrics2.append({'accuracy': [], 'iou': [], 'dice': [], 'precision': [], 'recall': []})  # creates a dictionary for each class

    t = time.time()
    model.eval()
    perf_accumulator = []
    loss = []
    for batch_idx, (data, target) in enumerate(test_loader):
        
        data, target = data.to(device), [i.to(device) for i in target]
        output = model(data)

        # metrics
        clssMetrics2, accuracy2, IoU2, dice2, precision2, recall2, superIOU, superACC, superPERF, superPREC, superRECA, levelLossTest = get_metrics(output, target, accuracy2, IoU2, dice2, precision2, recall2, superIOU, superACC, superPERF, superPREC, superRECA, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics2, args, True, lossFuncts, t_root, loss)
        

        # test loss
        if args.hierarchical_loss == True:
            lossTest = lossFuncts[0](output, target, t_root)[0].item() + lossFuncts[1](output, target, t_root)[0].item()
        else:
            lossTest = lossFuncts[0](output, target).item() + lossFuncts[1](output, target).item()


        # calculates perfect measure (dice) for all classes except background
        perf_accumulator.append(torch.mean(perf_measure(output, target, device, args.num_classes)[1][1:]).item())
        if batch_idx + 1 < len(test_loader):
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest  Epoch: {} [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    epoch,
                    batch_idx + 1,
                    len(test_loader),
                    100.0 * (batch_idx + 1) / len(test_loader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

        if args.save_images_batch == True and inital == True:
            if epoch % args.save_images_batch_num == 0 or epoch == 1:
                print('Saving batch images...')

                # gets the first image in the batch in the shape of (1,3,352,352)
                first_image = output[0,:,:,:].unsqueeze(0)
                # find probabilities of each class in output
                first_image = F.softmax(first_image, dim = 1)
                # find the most likely class in output
                first_image = torch.argmax(first_image, dim=1)
                # split output into a list of tensors, each tensor conains the pixels of a single class with 0 and 255 values
                first_image = performance_metrics.split_targets(first_image, args.num_classes, pos_class_val=255)
                # loops through each class and saves
                for j in range(len(first_image)):
                    predicted_map = np.squeeze(np.array(first_image[j].cpu()))
                    cv2.imwrite(os.path.join(save_loc, "images", str(j), "Epoch"+str(epoch)+".png"), predicted_map * 255)
                inital = False


    # means the contents of each key in each dictionary in clssMetrics
    for met in range(len(clssMetrics2)):
        for key in clssMetrics2[met]:
            clssMetrics2[met][key] = np.mean(clssMetrics2[met][key])

    print("FINISHED TESTING")
    return np.mean(perf_accumulator).item(), np.std(perf_accumulator).item(), clssMetrics2, np.mean(accuracy2).item(), np.mean(IoU2).item(), np.mean(dice2).item(), np.mean(precision2).item(), np.mean(recall2).item(), np.mean(superIOU).item(), np.mean(superACC).item(), np.mean(superPERF).item(), np.mean(superPREC).item(), np.mean(superRECA).item(), levelLossTest, lossTest

def build(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # creates hiararchy tree from file
    t_root = create_tree_from_textfile(args.tree_root)
    add_channels(t_root,0)
    add_levels(t_root,find_depth(t_root))

    img_path = os.path.join(args.root, "images", '*')
    input_paths = sorted(glob.glob(img_path))
    depth_path = os.path.join(args.root, "masks", '*') 
    target_paths = sorted(glob.glob(depth_path))
    
    # Additional code to use a pre split val dataset
    if args.val_dataset != 'None':
        img_path2 = os.path.join(args.val_dataset, "images", '*')
        depth_path2 = os.path.join(args.val_dataset, "masks", '*')
        val_img_path = sorted(glob.glob(img_path2))
        val_target_path = sorted(glob.glob(depth_path2))
    else:
        val_img_path = 'None'
        val_target_path = 'None'
    
    if not val_img_path:
        print('Val Set Is Empty')
        sys.exit()

    train_dataloader, _, val_dataloader = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=args.batch_size, val_batch_size=args.val_batch, val_img=val_img_path, val_target=val_target_path, test_img='None', test_target='None', img_size=args.img_size, test_remove=args.test_remove, num_classes=args.num_classes, workers_num=args.num_workers, include_backg=args.include_background
    )

    # initialise loss functions
    if args.hierarchical_loss == True:
        loss = [losses.TreeCrossEntropyLoss(), losses.TreeSoftDiceLoss(num_classes=args.num_classes)]
    else:
        loss = [losses.CrossEntropyLoss(), losses.SoftDiceLoss(num_classes=args.num_classes)]


    Accuracy = performance_metrics.Accuracy()
    Iou = performance_metrics.Jaccardindex()
    perf = performance_metrics.DiceScore()
    Precision = performance_metrics.Precision()
    Recall = performance_metrics.Recall()

    model = models.FCBFormer(size=args.img_size, weightPth=args.weight_path, n_classes=args.num_classes)

    # loads full model weights from .pt file
    if args.model_weights != 'None':
        load_state_dict = torch.load(os.path.join(args.model_weights))["model_state_dict"]

        if args.no_ph_weights == True:
            # removes weights from output layer in .pt file, to allow loading of model with different number of output nodes
            curr_state_dict = model.state_dict()
            load_state_dict = {k: v for k, v in load_state_dict.items() if "PH" not in k}
            curr_state_dict.update(load_state_dict)
            model.load_state_dict(curr_state_dict)
        else:
            model.load_state_dict(load_state_dict)

    if args.mgpu == "true":
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return (
        device,
        train_dataloader,
        val_dataloader,
        loss,
        Accuracy,
        Iou,
        perf,
        Precision,
        Recall,
        model,
        optimizer,
        t_root,
    )


def train(args):
    (
        device,
        train_dataloader,
        val_dataloader,
        lossFunc,
        Accuracy,
        Iou,
        perf,
        Precision,
        Recall,
        model,
        optimizer,
        t_root,
    ) = build(args)

    save_loc = args.save_path
    if os.path.exists(save_loc):
        # creates another folder of the same name + 1
        print("Save location already exists")
        runNumtemp = os.path.split(save_loc)[-1].split('_')
        folName = runNumtemp[0]
        runNum = 1
        while os.path.exists(save_loc+'_'+str(runNum)):
            runNum += 1
        save_loc = os.path.join(os.path.split(save_loc)[0], folName+ "_" + str(runNum))
    try:
        # Creates save folder
        os.makedirs(save_loc)
    except:
        pass

    if not os.path.exists(os.path.join(save_loc, "images")):
        os.makedirs(os.path.join(save_loc, "images"))
    
    for clss2 in range(args.num_classes):
        if not os.path.exists(os.path.join(save_loc, "images", str(clss2))):
            os.makedirs(os.path.join(save_loc, "images", str(clss2)))

    # if there is a .csv file in the save location, delete it
    if os.path.exists(os.path.join(save_loc, "metrics.csv")):
        os.remove(os.path.join(save_loc, "metrics.csv"))

    prev_best_test = None
    if args.lrs == "true":
        if args.lrs_min > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3,  min_lr=args.lrs_min, verbose=True
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=3, verbose=True
            )


    # START TRAINING EPOCHS
    for epoch in range(1, args.epochs + 1):
        # try:
        loss, trnClassMet, trnAcc, trnIoU, trnDice, trnPrecision, trnRecall, levelLoss = train_epoch(
            model, device, train_dataloader, optimizer, epoch, lossFunc, args, t_root, Accuracy, Iou, perf, Precision, Recall
        )
        test_measure_mean, test_measure_std, tstClassMet, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, superIOU, superACC, superPERF, superPREC, superRECA, levelLossTest, lossTest = test(
            model, device, val_dataloader, epoch, Accuracy, Iou, perf, Precision, Recall, args, save_loc, lossFunc, t_root
        )

        # save metrics in train save location
        if not os.path.exists(os.path.join(save_loc, "metrics.csv")):
            with open(os.path.join(save_loc, "metrics.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Train Loss", "Train Level Loss", "Train Accuracy", "Train IoU", "Train Dice", "Train Precision", "Train Recall", "Train Class Metrics", "Val Loss", "Val Level Loss", "Val Accuracy", "Val IoU", "Val Dice", "Val Precision", "Val Recall", "Val Test Measure Mean", "Val Test Measure Std", "Val Class Metrics"])
                writer.writerow([epoch, loss, levelLoss, trnAcc, trnIoU, trnDice, trnPrecision, trnRecall, trnClassMet, lossTest, levelLossTest, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, test_measure_mean, test_measure_std, tstClassMet])
        else:
            with open(os.path.join(save_loc, "metrics.csv"), 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, loss, levelLoss, trnAcc, trnIoU, trnDice, trnPrecision, trnRecall, trnClassMet, lossTest, levelLossTest, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, test_measure_mean, test_measure_std, tstClassMet])


        # print val metrics
        print('Validation Accuracy: ', tstAcc)
        print('Validation IoU: ', tstIoU)
        print('Validation Dice: ', tstDice)
        print('Validation Precision: ', tstPrecision)
        print('Validation Recall: ', tstRecall)

        # prints val metrics for each class
        for clss in range(len(tstClassMet)):
            print('Class: ', clss)
            print('Validation Accuracy: ', tstClassMet[clss]['accuracy'])
            print('Validation IoU: ', tstClassMet[clss]['iou'])
            print('Validation Dice: ', tstClassMet[clss]['dice'])
            print('Validation Precision: ', tstClassMet[clss]['precision'])
            print('Validation Recall: ', tstClassMet[clss]['recall'])
        
        if args.calc_super != False:
            print('Superclass Metrics class 1 and 2')
            print('Superclass Val Accuracy: ', superACC)
            print('Superclass Val IoU: ', superIOU)
            print('Superclass Val Dice: ', superPERF)
            print('Superclass Val Precision: ', superPREC)
            print('Superclass Val Recall: ', superRECA)

        if args.lrs == "true":
            scheduler.step(test_measure_mean)
        if prev_best_test == None or test_measure_mean > prev_best_test:
            print("Saving Best...")
            # save current model as temp best
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict()
                    if args.mgpu == "false"
                    else model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "test_measure_mean": test_measure_mean,
                    "test_measure_std": test_measure_std,
                },
                os.path.join(save_loc, "new_best.pt"),
            )
            # Delete old .PT
            if os.path.exists(os.path.join(save_loc, "best.pt")):
                os.remove(os.path.join(save_loc, "best.pt"))
            # change name of old .PT
            os.rename(os.path.join(save_loc, "new_best.pt"), os.path.join(save_loc, "best.pt"))

            print('SAVED BEST')
            prev_best_test = test_measure_mean
        print("Saving last...")

        # save current model as temp best
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict()
                if args.mgpu == "false"
                else model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "test_measure_mean": test_measure_mean,
                "test_measure_std": test_measure_std,
            },
            os.path.join(save_loc, "new_last.pt"),
        )
        # Delete old .PT
        if os.path.exists(os.path.join(save_loc, "last.pt")):
            os.remove(os.path.join(save_loc, "last.pt"))
        # change name of old .PT
        os.rename(os.path.join(save_loc, "new_last.pt"), os.path.join(save_loc, "last.pt"))
        print('SAVED LAST')



def get_args():
    #set 
    parser = argparse.ArgumentParser(description="Train FCBFormer on specified dataset")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--weight-path", type=str, required=True) # transformer pretrained weights (required for every run: even if model_weights are parsed)
    parser.add_argument("--model-weights", type=str, default='None') # load pretained weights from previous runs
    parser.add_argument("--no-ph-weights", type=str, default="True") # if True, does not load the predictor head (if you have a different shape output nodes)
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--tree-root", type=str, required=True) # full path to tree root file location
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-dataset", type=str, default="None")
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--learning-rate", type=float, default=1e-4, dest="lr") # 1e-4 1e-2
    parser.add_argument("--test-remove", type=str, default="True")
    parser.add_argument("--val-batch", type=int, default=1)
    parser.add_argument("--num-classes", type=int, required=True, default=1)
    parser.add_argument("--num-workers", type=int, default=-1) #-1 for auto workers
    parser.add_argument("--calc-super", type=str, default="False")
    parser.add_argument("--learning-rate-scheduler", type=str, default="true", dest="lrs")
    parser.add_argument("--learning-rate-scheduler-minimum", type=float, default=1e-6, dest="lrs_min") # 1e-6 1e-3
    parser.add_argument("--multi-gpu", type=str, default="false", dest="mgpu", choices=["true", "false"])
    parser.add_argument("--save-images-batch", type=str, default="False")
    parser.add_argument("--save-images-batch-num", type=int, default=10)
    parser.add_argument("--include-background", type=str, default="False")
    parser.add_argument("--hierarchical-loss", type=str, default="True")

    
    return parser.parse_args()


def main():
    args = get_args()

    # converts string to boolean for some args
    args.no_ph_weights = True if args.no_ph_weights == "True" else False
    args.test_remove = True if args.test_remove == "True" else False
    args.calc_super = True if args.calc_super == "True" else False
    args.save_images_batch = True if args.save_images_batch == "True" else False
    args.include_background = True if args.include_background == "True" else False
    args.hierarchical_loss = True if args.hierarchical_loss == "True" else False

    train(args)


if __name__ == "__main__":
    main()
    
