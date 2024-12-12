import torch
from torchvision import transforms
from PIL import Image
from torcheval.metrics import FrechetInceptionDistance
import os
import csv
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='FID Calculation',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data', type=str,
                    default='', help='Name of data set')
parser.add_argument('--num_class', type=int,
                    default=2, help='number of classes')
parser.add_argument('--output_scores', type=str,
                    default='', help='path of csv output for FID scores')
parser.add_argument('--attack_path', type=str,
                    default='', help='path to splits of output generated images')
parser.add_argument('--orig_path', type=str,
                    default='', help='path to splits of of real images to input')
args = parser.parse_args()

# Function to load and preprocess images
def preprocess_image(image_path, image_size=(299, 299)):
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),  # Converts image to tensor and scales to [0, 1]
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Function to load all images from a directory into a batch
# def load_images_from_dir(directory, image_size=(299, 299)):
#     images = []
#     for img_file in os.listdir(directory):
#         img_path = os.path.join(directory, img_file)
#         if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Ensure valid image files
#             images.append(preprocess_image(img_path, image_size))
#     return torch.cat(images, dim=0)  # Combine all images into a single batch

# # Paths to the folders containing real and generated images
def load_images_from_dir(image_dir):
    images = []
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Example resizing (adjust as needed)
        transforms.ToTensor(),          # Convert the image to a tensor
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])

    # Walk through the directory to load images
    for filename in os.listdir(image_dir):
        if filename.endswith(('png', 'jpg', 'jpeg', 'bmp')):  # Filter for valid image extensions
            image_path = os.path.join(image_dir, filename)
            # try:
            image = Image.open(image_path).convert('RGB')  # Open the image and convert to RGB
            
            images.append(transform(image).unsqueeze(0))  # Apply transformation and append to the list
            # except Exception as e:
            #     print(f"Error loading image {image_path}: {e}")
            #     continue

    # Check if the images list is empty
    if len(images) == 0:
        print(f"No valid images found in {image_dir}.")
        return None  # Optionally return an empty tensor with torch.empty(0)

    return torch.cat(images, dim=0)  # Concatenate all image tensors into a single batch


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

for classifier in ['squeezenet1.1','densenet121', 'resnet34']: #
    class_row = []
    for class_name in ["benign" , "malignant"]:
        split_row = []
        for split in [1, 2 , 3 , 4, 5]:
            real_images_dir = f"{args.orig_path}/{args.data}_splits/split_{split}/val/{class_name}"
            generated_images_dir = f"{args.attack_path}/{args.data}/{classifier}/split{split}/0.05/{class_name}"

            # Load and preprocess images from both folders
            real_images = load_images_from_dir(real_images_dir)
            generated_images = load_images_from_dir(generated_images_dir)

            # Check if images were loaded correctly
            if generated_images is None or real_images.size(0) == 0 or generated_images.size(0) == 0 or generated_images.size(0) == 1:
                #raise ValueError("One of the directories is empty or contains no valid images.")
                continue

            # Create a FrechetInceptionDistance object
            fid_metric = FrechetInceptionDistance(feature_dim=2048)

            # Update the FID metric with real and generated images
            fid_metric.update(real_images, is_real=True)     # Add real images
            fid_metric.update(generated_images, is_real=False)  # Add generated images

            # Compute the FID score
            fid_value = fid_metric.compute()
            split_row.append(fid_value)

            # Print the FID score
            print(f"FID score between the two sets of images: {fid_value.item()}")
        class_row.append(np.mean(split_row))
        
    row = [classifier, class_row[0], class_row[1], np.mean(class_row)]
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(row)
            
