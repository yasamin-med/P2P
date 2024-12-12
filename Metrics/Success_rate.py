import os
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Success Rate Calculation',
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

# Function to determine label and folder from image path
def get_label_and_folder_from_path(image_path):
    if "benign" in image_path:
        return 0, "benign"
    elif "malignant" in image_path:
        return 1, "malignant"

# Function to preprocess image (assuming all classifiers take 224x224 input size)
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Main function
def main():
    results = []
    
    orig_path_base = args.orig_path
    attack_path_base = args.attack_path

    for classifier_name in ["squeezenet1.1", "densenet121", "resnet34"]:
        for split in [1, 2, 3, 4, 5]:
            num_attack = 0
            total_images = 0

            # Counters for each class
            correct_benign = 0
            correct_malignant = 0
            # correct_normal = 0

            total_benign = 0
            total_malignant = 0
            # total_normal = 0

            # Load the appropriate model
            if classifier_name == 'squeezenet1.1':
                net = models.squeezenet1_1(pretrained=False)
                net.classifier[1] = nn.Conv2d(512, args.num_class, kernel_size=(1, 1), stride=(1, 1))
                net.load_state_dict(torch.load(f"{args.classifier_path}/result_split_{split}/squeezenet1.1/squeezenet1.1.pth", weights_only=False))
            elif classifier_name == 'densenet121':
                net = models.densenet121(pretrained=False)
                net.classifier = nn.Linear(net.classifier.in_features, args.num_class)
                net.load_state_dict(torch.load(f"{args.classifier_path}/result_split_{split}/densenet121/densenet121.pth", weights_only=False))
            elif classifier_name == 'resnet34':
                net = models.resnet34(pretrained=False)
                net.fc = nn.Linear(net.fc.in_features, args.num_class)
                net.load_state_dict(torch.load(f"{args.classifier_path}/result_split_{split}/resnet34/resnet34.pth", weights_only=False))

            # Set the model to evaluation mode and move it to GPU (if available)
            net.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net.to(device)

            orig_path = os.path.join(orig_path_base, f"split_{split}/val")
            attack_path = os.path.join(attack_path_base, f"{classifier_name}/split{split}/0.05")

            # Walk through the original image folder and process images
            for subdir, dirs, files in os.walk(orig_path):
                for file in files:
                    if file.endswith(('png', 'jpg', 'jpeg', 'bmp')):
                        
                        image_path = os.path.join(subdir, file)
                        label, folder = get_label_and_folder_from_path(image_path)
                        image_tensor = preprocess_image(image_path).to(device)



                        # Classify the image
                        with torch.no_grad():
                            output = net(image_tensor)
                            _, predicted = torch.max(output, 1)

                        # Check if the classification was correct
                        if predicted.item() == label:
                            total_images += 1

                            # Count correct classifications per class
                            # Count total images per class
                            if label == 0:
                                total_benign += 1
                            elif label == 1:
                                total_malignant += 1
                            # elif label == 2:
                            #     total_normal += 1
                            
                            # Construct the adversarial image path with the correct folder name (benign, malignant, normal)
                            image_name_without_extension = os.path.splitext(file)[0]
                            adv_image_name = f"{image_name_without_extension}_attacked.png"
                            
                            # adv_image_name = 
                            adv_image_path = os.path.join(attack_path, folder, adv_image_name)

                            # Check if the adversarial image exists
                            if os.path.exists(adv_image_path):
                                num_attack += 1
                                if label == 0:
                                    correct_benign += 1
                                elif label == 1:
                                    correct_malignant += 1
                                # elif label == 2:
                                #     correct_normal += 1

            # Calculate success rates
            success_rate = num_attack / total_images if total_images > 0 else 0
            benign_success_rate = correct_benign / total_benign if total_benign > 0 else 0
            malignant_success_rate = correct_malignant / total_malignant if total_malignant > 0 else 0
            # normal_success_rate = correct_normal / total_normal if total_normal > 0 else 0

            results.append({
                "Classifier": classifier_name,
                "Split": split,
                "Total Correct": total_images,
                "Total Benign": total_benign,
                "Correct Benign": correct_benign,
                "Benign Success Rate": benign_success_rate,
                "Total Malignant": total_malignant,
                "Correct Malignant": correct_malignant,
                "Malignant Success Rate": malignant_success_rate,
                # "Total Normal": total_normal,
                # "Correct Normal": correct_normal,
                # "Normal Success Rate": normal_success_rate,
                "Successful Attacks": num_attack,
                "Overall Success Rate": success_rate
            })

    # Save the results for each classifier and split to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(args.output_scores, index=False)

            #print(f"Classifier: {classifier_name}, Split: {split}, Success Rate: {success_rate}")

if __name__ == "__main__":
    main()
