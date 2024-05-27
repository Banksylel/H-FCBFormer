import torch
import os
import argparse
import time
import numpy as np
import glob
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from Data import dataloaders
from Models import models
from Metrics import performance_metrics

from train import get_metrics


def build(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pre_split_val == True:
        img_path = os.path.join(args.root, "images", '*')
        input_paths = sorted(glob.glob(img_path))
        depth_path = os.path.join(args.root, "masks", '*') 
        target_paths = sorted(glob.glob(depth_path))
        

    
    _, test_dataloader, _ = dataloaders.get_dataloaders(
        input_paths, target_paths, batch_size=1, img_size=args.img_size, types='Predict', num_classes=args.num_classes, workers_num=args.num_workers, include_backg=args.include_background
    )


    accuracy = performance_metrics.Accuracy()
    iou = performance_metrics.Jaccardindex()
    perf = performance_metrics.DiceScore()
    precision = performance_metrics.Precision()
    recall = performance_metrics.Recall()

    model = models.FCBFormer(size=args.img_size, weightPth=args.pretrain_weights, n_classes=args.num_classes)

    state_dict = torch.load(
        os.path.join(args.model_weights)
    )
    model.load_state_dict(state_dict["model_state_dict"])

    model.to(device)

    return device, test_dataloader, accuracy, iou, perf, precision, recall, model, target_paths


@torch.no_grad()
def predict(args):
    device, test_dataloader, Accuracy, Iou, perf_measure, Precision, Recall, model, target_paths = build(args)

    if args.save_images:
        if not os.path.exists("./Predictions"):
            os.makedirs("./Predictions")
        if not os.path.exists("./Predictions/{}".format(args.train_dataset)):
            os.makedirs("./Predictions/{}".format(args.train_dataset))
        for items in range(args.num_classes):
            if not os.path.exists("./Predictions/{}/{}".format(args.train_dataset, items)):
                os.makedirs("./Predictions/{}/{}".format(args.train_dataset, items))



    t = time.time()
    model.eval()
    perf_accumulator = []

    IoU2 = []
    dice2 = []
    precision2 = []
    recall2 = []
    accuracy2 = []
    superIOU, superACC, superPERF, superPREC, superRECA = [], [], [], [], []

    clssMetrics2 = []

    # sets up dictionary for class metrics
    for clss in range(args.num_classes):
        # creates a dictionary in classMetrics
        clssMetrics2.append({'accuracy': [], 'iou': [], 'dice': [], 'precision': [], 'recall': []})


    for i, (data, target) in enumerate(test_dataloader):
        data, target = data.to(device), [i.to(device) for i in target]
        output = model(data)

        # calculates metrics
        tstClassMet, tstAcc, tstIoU, tstDice, tstPrecision, tstRecall, superIOU, superACC, superPERF, superPREC, superRECA, _ = get_metrics(output, target, accuracy2, IoU2, dice2, precision2, recall2, superIOU, superACC, superPERF, superPREC, superRECA, Accuracy, Iou, perf_measure, Precision, Recall, device, clssMetrics2, args, False, None, None, None)        

        perf_accumulator.append(torch.mean(perf_measure(output, target, device, args.num_classes)[1][1:]).item())

        if i + 1 < len(test_dataloader):
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                ),
                end="",
            )
        else:
            print(
                "\rTest: [{}/{} ({:.1f}%)]\tAverage performance: {:.6f}\tTime: {:.6f}".format(
                    i + 1,
                    len(test_dataloader),
                    100.0 * (i + 1) / len(test_dataloader),
                    np.mean(perf_accumulator),
                    time.time() - t,
                )
            )

        # find probabilities of each class in output
        output = F.softmax(output, dim = 1)
        # find the most likely class in output
        output = torch.argmax(output, dim=1)
        # split output into a list of tensors, each tensor conains the pixels of a single class with 0 and 255 values
        outputs = performance_metrics.split_targets(output, args.num_classes, pos_class_val=255)
        # loops through each class and saves
        for j in range(len(outputs)):
            predicted_map = np.squeeze(np.array(outputs[j].cpu()))
            cv2.imwrite("./Predictions/{}/{}/{}".format(args.train_dataset, str(j), os.path.basename(target_paths[i])), predicted_map * 255)

    
    

    print("FINISHED TESTING")


    # print val metrics
    print('Validation Accuracy: ', np.mean(tstAcc).item(), "   (", np.std(tstAcc).item(), ")  ")
    print('Validation IoU: ', np.mean(tstIoU).item(), "   (", np.std(tstIoU).item(), ")  ")
    print('Validation Dice: ', np.mean(tstDice).item(), "   (", np.std(tstDice).item(), ")  ")
    print('Validation Precision: ', np.mean(tstPrecision).item(), "   (", np.std(tstPrecision).item(), ")  ")
    print('Validation Recall: ', np.mean(tstRecall).item(), "   (", np.std(tstRecall).item(), ")  ")




    # prints val metrics for each class
    for clss in range(len(tstClassMet)):
        print('Class: ', clss)
        print('Validation Accuracy: ', np.mean(np.array(tstClassMet[clss]['accuracy'])).item(), "   (", np.std(np.array(tstClassMet[clss]['accuracy'])).item(), ")  ")
        print('Validation IoU: ', np.mean(np.array(tstClassMet[clss]['iou'])).item(), "   (", np.std(np.array(tstClassMet[clss]['iou'])).item(), ")  ")
        print('Validation Dice: ', np.mean(np.array(tstClassMet[clss]['dice'])).item(), "   (", np.std(np.array(tstClassMet[clss]['dice'])).item(), ")  ")
        print('Validation Precision: ', np.mean(np.array(tstClassMet[clss]['precision'])).item(), "   (", np.std(np.array(tstClassMet[clss]['precision'])).item(), ")  ")
        print('Validation Recall: ', np.mean(np.array(tstClassMet[clss]['recall'])).item(), "   (", np.std(np.array(tstClassMet[clss]['recall'])).item(), ")  ")
    
    if args.calc_super != False:
        print('Superclass Metrics class 1 and 2')
        print('Superclass Val Accuracy: ', np.mean(superACC).item(), "   (", np.std(superACC).item(), ")  ")
        print('Superclass Val IoU: ', np.mean(superIOU).item(), "   (", np.std(superIOU).item(), ")  ")
        print('Superclass Val Dice: ', np.mean(superPERF).item(), "   (", np.std(superPERF).item(), ")  ")
        print('Superclass Val Precision: ', np.mean(superPREC).item(), "   (", np.std(superPREC).item(), ")  ")
        print('Superclass Val Recall: ', np.mean(superRECA).item(), "   (", np.std(superRECA).item(), ")  ")



    # save metrics to txt file in save location
    with open("./Predictions/{}/metrics.txt".format(args.train_dataset), "w") as file:
        file.write("Validation Accuracy: " + str(np.mean(tstAcc).item()) + "   (" + str(np.std(tstAcc).item()) + ")\n")
        file.write("Validation IoU: " + str(np.mean(tstIoU).item()) + "   (" + str(np.std(tstIoU).item()) + ")\n")
        file.write("Validation Dice: " + str(np.mean(tstDice).item()) + "   (" + str(np.std(tstDice).item()) + ")\n")
        file.write("Validation Precision: " + str(np.mean(tstPrecision).item()) + "   (" + str(np.std(tstPrecision).item()) + ")\n")
        file.write("Validation Recall: " + str(np.mean(tstRecall).item()) + "   (" + str(np.std(tstRecall).item()) + ")\n")
        file.write("\n")
        file.write("\n")
        file.write("Class Metrics: \n")
        for clss in range(len(tstClassMet)):
            file.write("Class: " + str(clss) + "\n")
            file.write("Validation Accuracy: " + str(np.mean(np.array(tstClassMet[clss]['accuracy'])).item()) + "   (" + str(np.std(np.array(tstClassMet[clss]['accuracy'])).item()) + ")\n")
            file.write("Validation IoU: " + str(np.mean(np.array(tstClassMet[clss]['iou'])).item()) + "   (" + str(np.std(np.array(tstClassMet[clss]['iou'])).item()) + ")\n")
            file.write("Validation Dice: " + str(np.mean(np.array(tstClassMet[clss]['dice'])).item()) + "   (" + str(np.std(np.array(tstClassMet[clss]['dice'])).item()) + ")\n")
            file.write("Validation Precision: " + str(np.mean(np.array(tstClassMet[clss]['precision'])).item()) + "   (" + str(np.std(np.array(tstClassMet[clss]['precision'])).item()) + ")\n")
            file.write("Validation Recall: " + str(np.mean(np.array(tstClassMet[clss]['recall'])).item()) + "   (" + str(np.std(np.array(tstClassMet[clss]['recall'])).item()) + ")\n")
            file.write("\n")
        
        if args.calc_super != False:
            file.write("\n")
            file.write("\n")
            file.write("Superclass Metrics class 1 and 2\n")
            file.write("Superclass Val Accuracy: " + str(np.mean(superACC).item()) + "   (" + str(np.std(superACC).item()) + ")\n")
            file.write("Superclass Val IoU: " + str(np.mean(superIOU).item()) + "   (" + str(np.std(superIOU).item()) + ")\n")
            file.write("Superclass Val Dice: " + str(np.mean(superPERF).item()) + "   (" + str(np.std(superPERF).item()) + ")\n")
            file.write("Superclass Val Precision: " + str(np.mean(superPREC).item()) + "   (" + str(np.std(superPREC).item()) + ")\n")
            file.write("Superclass Val Recall: " + str(np.mean(superRECA).item()) + "   (" + str(np.std(superRECA).item()) + ")\n")
        
            


def get_args():
    parser = argparse.ArgumentParser(
        description="Make predictions on specified dataset"
    )
    parser.add_argument("--train-dataset", type=str, required=True) # the save name for the predictions
    parser.add_argument("--data-root", type=str, required=True, dest="root")
    parser.add_argument("--full-ds", type=str, default="False")
    parser.add_argument("--pre-split-val", type=str, required=True, default="False")
    parser.add_argument("--model-weights", type=str, required=True)
    parser.add_argument("--pretrain-weights", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=352)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--save-images", type=str, default="True")
    parser.add_argument("--calc-super", type=str, default="False")
    parser.add_argument("--include-background", type=str, default="False")
    parser.add_argument("--include-std-div", type=str, default="False")
    

    return parser.parse_args()


def main():
    args = get_args()

    # converts string to boolean for some args 
    args.full_ds = True if args.full_ds == "True" else False
    args.pre_split_val = True if args.pre_split_val == "True" else False
    args.save_images = True if args.save_images == "True" else False
    args.calc_super = True if args.calc_super == "True" else False
    args.include_background = True if args.include_background == "True" else False
    args.include_std_div = True if args.include_std_div == "True" else False


    predict(args)


if __name__ == "__main__":
    main()

