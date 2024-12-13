import argparse
import time
import os

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.models as models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix , f1_score
import csv

def list_of_strings(arg):
    return arg.split(',')

parser = argparse.ArgumentParser(description='Train Dataset on various nets on medical datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--data_path', type=str,
                    default='./data/', help='Path of dataset for train ')
parser.add_argument('--data_test_path', type=str,
                    default='./data/', help='Path of dataset for test')
parser.add_argument('--data_valid_path', type=str,
                    default='./data/', help='Path of dataset for validation')
parser.add_argument('--output_path', type=str,
                    default='out', help='Path of output')
parser.add_argument('--output_file_name', type=str,
                    default='result', help='name of the output file')
parser.add_argument('--adjective_flag', type= int,
                    default= 1, help=' to evaluate over adjective list else goes for original')
parser.add_argument('--batch_size', type=int,
                    default=16, help='batch size')
parser.add_argument('--num_class', type=int,
                    default=2, help='number of classes')
parser.add_argument('--lr', type=float,
                    default= 0.001, help='Learning rate')
parser.add_argument('--num_epochs', type=int,
                    default= 100, help='epoch number')
parser.add_argument('--adjective_list', type=list_of_strings , default= [] , help ='list of adjectives')
parser.add_argument('--baselines', type=list_of_strings , default= ['densenet121', 'resnet34','convnext_base'] , help ='list of baselines' )
parser.add_argument('--train', type=int , default= 1 , help ='if 1 train and test if 0 just test' )
parser.add_argument('--size', type=int , default= 224 , help ='input size' )
parser.add_argument('--num_class_test', type=int,
                    default=3, help='number of classes for test')


args = parser.parse_args()
print(args)





# /////////////// Model Setup ///////////////

def get_net(model_name):
    if model_name == 'alexnet':
        weights = models.AlexNet_Weights.IMAGENET1K_V1
        net = models.alexnet(weights=weights)

    elif model_name == 'squeezenet1.0':
        weights = models.SqueezeNet1_0_Weights.IMAGENET1K_V1
        net = models.squeezenet1_0(weights=weights)


    elif model_name == 'squeezenet1.1':
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1
        net = models.squeezenet1_1(weights=weights)
        net.classifier[1] = nn.Conv2d(512, args.num_class, kernel_size=(1, 1), stride=(1, 1))


    elif model_name == 'vgg11':
        weights = models.VGG11_Weights.IMAGENET1K_V1
        net = models.vgg11(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, args.num_class)
 

    elif model_name == 'vgg19':
        weights = models.VGG19_Weights.IMAGENET1K_V1
        net = models.vgg19(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, args.num_class)
      
    
    elif model_name == 'vggbn':
        weights = models.VGG19_BN_Weights.IMAGENET1K_V1
        net = models.vgg19_bn(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, args.num_class)
        

    elif model_name == 'densenet121':
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        net = models.densenet121(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, args.num_class)

    elif model_name == 'densenet169':
        weights = models.DenseNet169_Weights.IMAGENET1K_V1
        net = models.densenet169(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, args.num_class)

    elif model_name == 'densenet201':
        weights = models.DenseNet201_Weights.IMAGENET1K_V1
        net = models.densenet201(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, args.num_class)

    elif model_name == 'densenet161':
        weights = models.DenseNet161_Weights.IMAGENET1K_V1
        net = models.densenet161(weights=weights)
        net.classifier = nn.Linear(net.classifier.in_features, args.num_class)

    elif model_name == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)

    elif model_name == 'resnet34':
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        net = models.resnet34(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)

    elif model_name == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        net = models.resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)

    elif model_name == 'resnet50_stylized': # not modifies yet
        # model_url = 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar'
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        net = models.resnet50()
        checkpoint = torch.load('cache/models/checkpoints/resnet50-stylized.pth.tar')
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint["state_dict"])
    
        

    elif model_name == 'resnet50_augmix':# not modifies yet
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        net = models.resnet50()
        checkpoint = torch.load('cache/models/checkpoints/resnet50-augmix.pth.tar')
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['state_dict'])
        net.fc = nn.Linear(net.fc.in_features, args.num_class)

    elif model_name == 'resnet101':
        weights = models.ResNet101_Weights.IMAGENET1K_V2
        net = models.resnet101(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)
        

    elif model_name == 'resnet152':
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        net = models.resnet152(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)

    elif model_name == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        net = models.vit_b_16(weights=weights)
        net.heads = nn.Linear(net.heads.head.in_features, args.num_class)

    elif model_name == 'vit_b_32':
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
        net = models.vit_b_32(weights=weights)
        net.heads = nn.Linear(net.heads.head.in_features, args.num_class)

    elif model_name == 'vit_l_16':
        weights = models.ViT_L_16_Weights.IMAGENET1K_V1
        net = models.vit_l_16(weights=weights)
        net.heads = nn.Linear(net.heads.head.in_features, args.num_class)

    elif model_name == 'vit_l_32':
        weights = models.ViT_L_32_Weights.IMAGENET1K_V1
        net = models.vit_l_32(weights=weights)
        net.heads = nn.Linear(net.heads.head.in_features, args.num_class)

    elif model_name == 'convnext_base':
        weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
        net = models.convnext_base(weights=weights)
        in_features = net.classifier[-1].in_features
        net.classifier[2] = nn.Linear(in_features, args.num_class)
       

    elif model_name == 'swin_b':
        weights = models.Swin_B_Weights.IMAGENET1K_V1
        net = models.swin_b(weights=weights)
        net.num_classes = args.num_class
        net.head = nn.Linear(net.head.in_features ,args.num_class)
        

    elif model_name == 'swin_v2_b':
        weights = models.Swin_V2_B_Weights.IMAGENET1K_V1
        net = models.swin_v2_b(weights=weights)
        net.num_classes = args.num_class
        net.head = nn.Linear(net.head.in_features, args.num_class)

    elif model_name == 'resnext50':
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        net = models.resnext50_32x4d(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)

    elif model_name == 'resnext101':
        weights = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        net = models.resnext101_32x8d(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args.num_class)

    elif model_name == 'resnext101_64':
        weights = models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1
        net = models.resnext101_64x4d(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, args)
    elif model_name == "efficientnet":
        net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        net.classifier.fc = nn.Linear(net.classifier.fc.in_features, args.num_class)      
    elif model_name == "inceptionnet":
        net = models.inception_v3(pretrained=True, aux_logits=True)
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, args.num_class)

        # Adjusting the auxiliary classifier
        num_ftrs_aux = net.AuxLogits.fc.in_features
        net.AuxLogits.fc = nn.Linear(num_ftrs_aux, args.num_class)
    args.prefetch = 4

    # for p in net.parameters():
    #     p.volatile = True

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    
    device = torch.device("cuda" )
    net.to(device)    

    #preprocess = weights.transforms()

    print(f'Model {model_name} Loaded')

    return net

# /////////////// Data Loader ///////////////

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization values for ResNet
    ])
device = torch.device("cuda" )



# /////////////// Further Setup ///////////////




def evaluate(net, model_name , adj):
    
    if os.path.isdir( args.output_path )== False:
            os.mkdir( args.output_path )
        
    if os.path.isdir(os.path.join( args.output_path ,adj ))== False:
        os.mkdir(os.path.join( args.output_path , adj ))
        
    if os.path.isdir(os.path.join( args.output_path ,adj, model_name ))== False:
        os.mkdir(os.path.join( args.output_path , adj , model_name ))
        
    if os.path.isfile(os.path.join(args.output_path , adj ,model_name, args.output_file_name +".txt")) == False:
        f = open(os.path.join(args.output_path ,adj, model_name , args.output_file_name + ".txt"), "x")
    f = open(os.path.join(args.output_path , adj, model_name , args.output_file_name + ".txt"), "w")
    #f.write('\n'.join(args))
    f.write(model_name +" Starts")

    if args.train == 1:
        # Create datasets for train, validation, and test
        if args.adjective_flag == 1:
            train_path = os.path.join( args.data_path , adj , "train" )
        else:
            train_path = args.data_path
        validation_path = args.data_valid_path
        
        
        train_dataset = ImageFolder(train_path, transform=transform)
        val_dataset = ImageFolder(validation_path, transform=transform)
        


        # Define data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True ,  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        

        
        # set optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), args.lr )
        # structure of output ---> output_path/adj/model_name ----> .pth and result.txt
        

    
        acc_best = 0
        for epoch in range(args.num_epochs):
            net.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                if model_name == "inceptionnet":
                    
                    outputs, aux_outputs = net(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux_outputs, labels)
                    loss = loss1 + 0.4*loss2
                else:
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)                

                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Print the average loss for this epoch
            print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss/len(train_loader)}")
            f.write(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {running_loss/len(train_loader)}\n")
            # Validation loop
            net.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = net(inputs)
                    if model_name == "inceptionnet":
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            # Print the validation accuracy for this epoch
            print(f"Validation Accuracy: {100 * correct / total}%")
            f.write(f"Validation Accuracy: {100 * correct / total}%\n")
            if correct / total > acc_best:
                if os.path.isdir(os.path.join( args.output_path , adj , model_name ))== False:
                    os.mkdir(os.path.join( args.output_path,adj , model_name )) 
                acc_best = correct / total
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save(best_model_wts, os.path.join( args.output_path , adj , model_name , model_name + ".pth" ))
                
            
        print("Training complete on " + model_name)
        f.write("Training complete on " + model_name + "\n")
        print("best accuracy for validation")
        print(acc_best)
        f.write(f"best accuracy for validation: {acc_best}\n")
    
    ##### test
    test_path = args.data_test_path
    test_dataset = ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    net_best = net
    net_best.load_state_dict(torch.load(os.path.join( args.output_path ,adj, model_name , model_name + ".pth" ) ))
    net_best.eval()
    correct = 0
    total = 0
    confusion_matrix = torch.zeros(args.num_class_test, args.num_class_test)
    label_list = []
    predicted_list = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_best(inputs)
            if model_name == "inceptionnet":
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            _, predicted = torch.max(outputs, 1)
            predicted_list.extend(predicted.cpu().numpy())
            label_list.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
                
    test_accuracy = 100 * correct / total
    print(f"Test Accuracy: {test_accuracy}%")
    f.write(f"Test Accuracy: {test_accuracy}%\n")
    f1 = f1_score(label_list, predicted_list, average='macro')
    print(f"F1 score: {f1}%")
    f.write(f"F1 score: {f1}%\n")


    # Calculate specificity and sensitivity for each class
    
    sensitivity_avg = 0.0
    specifity_avg = 0.0
    precision_avg = 0.0
    for i in range(args.num_class_test):
        TP_i = confusion_matrix[i, i].item()
        FN_i = confusion_matrix[i, :].sum().item() - TP_i
        FP_i = confusion_matrix[:, i].sum().item() - TP_i
        TN_i = confusion_matrix.sum().item() - TP_i - FN_i - FP_i
        
        sensitivity_i = TP_i / (TP_i + FN_i + 1e-6)
        specificity_i = TN_i / (TN_i + FP_i + 1e-6)
        precision_i = TP_i / (TP_i + FP_i + 1e-6 )
        sensitivity_avg = sensitivity_avg + sensitivity_i / args.num_class_test
        specifity_avg = specifity_avg + specificity_i / args.num_class_test
        precision_avg = precision_avg + precision_i / args.num_class_test
        
        
        print(f"Class {i} - Sensitivity: {sensitivity_i:.4f}, Specificity: {specificity_i:.4f} , precision: {precision_i:.4f}")
        f.write(f"Class {i} - Sensitivity: {sensitivity_i:.4f}, Specificity: {specificity_i:.4f}, precision: {precision_i:.4f}\n")
    f.write(f"sensitivity_avg = {sensitivity_avg:.4f} , specifity_avg = {specifity_avg:.4f}, precision_avg = {precision_avg:.4f}, F1 = {f1: .4f}\n")
    print(f"sensitivity_avg = {sensitivity_avg:.4f} , specifity_avg = {specifity_avg:.4f}, precision_avg = {precision_avg:.4f}, F1 = {f1: .4f}")
    f.close()
    return precision_avg , sensitivity_avg , specifity_avg , test_accuracy, f1



# /////////////// End Further Setup ///////////////


# /////////////// Display Results ///////////////
#adjective_list = [ "colorful", "stylized" ,"high-contrast", "low-contrast", "posterized", "solarized", "sheared", "bright", "dark"]


'''baselines = ['vgg19', 'alexnet', 'squeezenet1.0', 'squeezenet1.1',
             'vgg11', 'vgg19', 'vggbn',
             'densenet121', 'densenet169', 'densenet201',
             'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
             'resnext50', 'resnext101', 'resnext101_64','vit_b_16', 'vit_b_32','vit_l_16' , 'vit_l_32','convnext_base',]'''            
baselines = args.baselines
if args.adjective_flag == 1:
    table_data = [['model name' ,'adj' , 'accuracy' , 'sensitivity' , 'specifity' , 'precision' , 'F1']]
    accuracies = np.zeros([len(baselines), len(args.adjective_list) + 1])
    for i, model in enumerate(baselines):
        model_name = model
        net = get_net(model)
        model_accuracies = []
        sensitivity_list = []
        specifity_list = []
        precision_list = []
        F1_list = []
        for adj in args.adjective_list:
            precision_avg , sensitivity_avg , specifity_avg , test_accuracy , f1 = evaluate(net , model_name, adj)
            model_accuracies.append(test_accuracy)
            sensitivity_list.append(sensitivity_avg)
            specifity_list.append(specifity_avg)
            precision_list.append(precision_avg)
            F1_list.append(f1)
            table_data.append([model_name , adj , test_accuracy , sensitivity_avg , specifity_avg , precision_avg , f1 ])

    # Specify the CSV file name
    csv_file_name = os.path.join( args.output_path , args.output_file_name + '_table.csv')

    # Writing the table to a CSV file
    with open(csv_file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(table_data)

    print(f'Table has been written to {csv_file_name}')
else:
    table_data = [['model name' ,'adj' , 'accuracy' , 'sensitivity' , 'specifity' , 'precision' , 'F1']]
    for i, model in enumerate(baselines):
        model_name = model
        net = get_net(model)

        precision_avg , sensitivity_avg , specifity_avg , test_accuracy , f1 = evaluate(net , model_name , "")
        table_data.append([model_name , '' , test_accuracy , sensitivity_avg , specifity_avg , precision_avg , f1 ])

    # Specify the CSV file name
    csv_file_name = os.path.join( args.output_path , args.output_file_name + '_table.csv')

    # Writing the table to a CSV file
    with open(csv_file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(table_data)

    print(f'Table has been written to {csv_file_name}') 
