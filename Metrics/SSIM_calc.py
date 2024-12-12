import sys


import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import csv
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import argparse


parser = argparse.ArgumentParser(description='SSIM Calculation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str,
                    default='', help='Name of data set')
parser.add_argument('--num_class', type=int,
                    default=2, help='number of classes')
parser.add_argument('--attack_path', type=str,
                    default='', help='path to splits of output generated images')
parser.add_argument('--orig_path', type=str,
                    default='', help='path to splits of of real images to input')
parser.add_argument('--output_scores', type=str,
                    default='', help='path of csv output for FID scores')
parser.add_argument('--classifier_path', type=str,
                    default='', help='path of splits for densenet121, resnet34, squeezenet1.1 classifiers')
args = parser.parse_args()

# Initialize the LPIPS loss function
# loss_fn = lpips.LPIPS(net='alex')

# Define image transformation
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])

# Function to load images and preprocess them
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    return image.unsqueeze(0)

# Function to load the classifier model
def load_classifier(name, classifier_path):
    if name == 'squeezenet1.1':
        net = models.squeezenet1_1(pretrained=False)
        net.classifier[1] = nn.Conv2d(512, args.num_class, kernel_size=(1, 1), stride=(1, 1))
        net.load_state_dict(torch.load(classifier_path + "/squeezenet1.1/squeezenet1.1.pth", weights_only=False))
    elif name == 'densenet121':
        net = models.densenet121(pretrained=False)
        net.classifier = nn.Linear(net.classifier.in_features, args.num_class)
        net.load_state_dict(torch.load(classifier_path + "/densenet121/densenet121.pth", weights_only=False))
    elif name == 'resnet34':
        net = models.resnet34(pretrained=False)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)
        net.load_state_dict(torch.load(classifier_path + "/resnet34/resnet34.pth", weights_only=False))
    net.eval()
    return net

# Classifier transformation
transform_classifier = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Base paths
orig_path_base = args.orig_path
attack_path_base = args.attack_path
csv_file_path = args.output_scores

# Ensure CSV directory exists
csv_file_dir = os.path.dirname(csv_file_path)
if not os.path.exists(csv_file_dir):
    os.makedirs(csv_file_dir)

# Initialize CSV file with header if it doesn't exist
with open(csv_file_path, 'a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if csvfile.tell() == 0:
        csvwriter.writerow(["classifier", "benign", "malignant", "average"])

# Iterate over classifiers
for classifier in ["squeezenet1.1" , 'resnet34' , 'densenet121']:
    class_row = []
    
    for class_name in ["benign", "malignant"]:
        label = torch.tensor([0]) if class_name == "benign" else torch.tensor([1]) if class_name == "malignant" else torch.tensor([2])
        
        split_row = []
        for split in ["1", "2", "3", "4", "5"]:
            classifier_path = f"{args.classifier_path}/result_split_{split}"
            net = load_classifier(classifier, classifier_path)
            
            orig_path = os.path.join(orig_path_base, f"split_{split}", "val", class_name)
            attack_path = os.path.join(attack_path_base, classifier, f"split{split}","0.05", class_name)
            list_image_orig = os.listdir(orig_path)

            total_ssim = 0.0
            for name in list_image_orig:
                image_orig_path = os.path.join(orig_path, name)
                name, extension = os.path.splitext(name)
                image_attack_path = os.path.join(attack_path, f"{name}_attacked.png")
                
                
                # Check if the output of the model is correct
                img_orig_classifier = transform_classifier(Image.open(image_orig_path).convert('RGB')).unsqueeze(0)
                out_classifier = net(img_orig_classifier)
                _, class_orig = torch.max(out_classifier , 1 )
                if not os.path.exists(image_attack_path):
                    continue

                if class_orig == label:
                    print(class_orig)
                    
                    image_orig = load_and_preprocess_image(image_orig_path)
                    image_attack = load_and_preprocess_image(image_attack_path)
                    ssim_value = ssim( image_orig, image_attack, data_range=255, size_average=False)
                    print(ssim_value)
                    total_ssim += ssim_value.item()
            
            split_row.append(total_ssim / len(list_image_orig) if len(list_image_orig) > 0 else 0.0)
        
        class_row.append(np.mean(split_row) if split_row else 0.0)
    
    avg_ssim = np.mean(class_row) if class_row else 0.0
    row = [classifier, class_row[0], class_row[1], avg_ssim]
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)
        
    print(f"avg ssim for {classifier}: {avg_ssim}")
