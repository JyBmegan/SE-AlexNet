import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from se_model import AlexNetWithSE, save_model_structure
from mydataset import affectnet

from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Automatically transfer weights for different condition
def load_weights(model, weight_path, device):
    pretrained_dict = torch.load(weight_path, map_location=device)
    if 'model' in pretrained_dict:
        pretrained_dict = pretrained_dict['model'] 
    model_dict = model.state_dict()
    new_state_dict = {}
    pretrained_conv_weights = []
    pretrianed_conv_biased = []
    # get all weights of Conv2d in pretrained model
    sorted_keys = sorted(pretrained_dict.keys(), key = lambda x: int(x.split('.')[1]) if len(x.split('x'))>1 and x.split('.')[1].isdigit() else 999)
    for key in sorted_keys:
        # weights for Conv
        if "features" in key and "weight" in key and len (pretrained_dict[key].shape) == 4:
            pretrained_conv_weights.append(pretrained_dict[key])
            bias_key = key.replace("weight", "bias")
            if bias_key in pretrained_dict:
                pretrianed_conv_biased.append(pretrained_dict[bias_key])
            else:
                pretrianed_conv_biased.append(None)
    print(f"Found {len(pretrained_conv_weights)} Conv Layers in pretrained files.")

    conv_idx = 0
    for name, module in model.features.named_children():
        # if it is conv2d
        if isinstance(module, nn.Conv2d):
            if conv_idx >= len(pretrained_conv_weights):
                print("Warning: More conv layers in model than in pretrained weights")
                break
            # weights
            module.weight.data = pretrained_conv_weights[conv_idx].to(device)
            if module.bias is not None and pretrianed_conv_biased[conv_idx] is not None:
                module.bias.data = pretrianed_conv_biased[conv_idx].to(device)
            print(f"Conv2d layer {name} is loaded from pretrained index {conv_idx}")
            conv_idx += 1
        elif isinstance(module, nn.Linear):
            pass

def matplot_loss(train_loss, val_loss, save_path = './loss.jpg'):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Comparison of loss values between training set and validation set")
    plt.savefig(save_path)
    plt.close()

def matplot_acc(train_acc, val_acc, save_path = './acc.jpg'):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title("Comparison of acc values between training set and validation set")
    plt.savefig(save_path)

def main():

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []

    batch_size = 128
    learning_rate = 0.0001
    epochs = 40
    
    resume = False

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_dataset = affectnet(dataset_dir=r'/home/zhang/share/home/scz6112/AffectNet/train_set/images',
                                         csv_path=r'/home/zhang/share/home/scz6112/AffectNet/AffectNet/modified_train_28000_sc.csv')
    validate_dataset = affectnet(dataset_dir=r'/home/zhang/share/home/scz6112/AffectNet/val_set/images',
                                         csv_path=r'/home/zhang/share/home/scz6112/AffectNet/AffectNet/modified_val_1000_sc.csv')
    

    # a_csv_path = r'/home/zhang/share/home/scz6112/AffectNet/AffectNet/modified_train_28000_sc.csv'
    # df_debug = pd.read_csv(a_csv_path)
    # counts = df_debug['expression'].value_counts().sort_index()
    # print(counts)

    val_num = len(validate_dataset)
    train_num = len(train_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)


    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    parser = argparse.ArgumentParser()
    parser.add_argument('--se_pos', type = int, help = '1-4')
    parser.add_argument('--reduction', type = int, help = '2/4/8/16/32')
    parser.add_argument('--trans', type = str, default = 'Face', help = 'Face or Object')
    args = parser.parse_args()

    current_se_pos = args.se_pos
    squeeze_ratio = args.reduction
    trans_con = args.trans

    # because of the orginal weight path, we have to do so
    if trans_con == 'Face':
        trans_path = 'face'
    elif trans_con == 'Object':
        trans_path = 'object'
    else:
        raise ValueError(f"Unknow Pretrain Type: {trans_con}")
    
    print (f"Training: Pos = {current_se_pos}, Ratio = {squeeze_ratio}, TransCondition = {trans_con}, OriginalWeightPath = {trans_path}")

    # # If you want to train and change the condition by yourself, uncommand these lines
    # current_se_pos = 1 # check: insert location in convolutional layer
    # squeeze_ratio = 4 # Check point
    # trans_con = "Face" # Check point
    # trans_path = "face" # Check point
    
    exp_name = f"SeC{current_se_pos}_{trans_con}Based_squeeze{squeeze_ratio}"

    net = AlexNetWithSE (num_classes=8, se_pos = current_se_pos)
    net.to(device)

    output_dir = f'/home/zhang/share/home/scz6112/AffectNet/ConvResults/{exp_name}' 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print (AlexNetWithSE().to(device))
    
    # print and save model structure
    save_model_structure(net, output_dir, filename = "current_model_arch.csv")

    model_weight_path = f'/media/zhang/97e9fbd4-1a76-43b2-a56c-570c3f238fa9/yfLi/project-paper3/weights/{trans_path}/AlexNet.pth' 

    load_weights(net, model_weight_path, device)

    # Configure frozen layer
    for name, child in net.features.named_children():
        if "SEBlock" in str(type(child)):
            print(f"Unfreezing {name} (SEBlock)")
            for param in child.parameters():
                param.requires_grad = True
        elif isinstance(child, nn.Conv2d):
            for param in child.parameters():
                param.requires_grad = False
    

    # pre_weights = torch.load(model_weight_path)
    # pre_dict = {k: v for k, v in pre_weights.items() if "classifier" not in k}
    # print(pre_dict)
    # missing_keys,unexpected_keys = net.load_state_dict(pre_dict, strict=False)
    # print(net)
    # net.classifier._modules['5'] = nn.Linear(4096, 11)  #revise 4096 to 1000 to match last nn.linear
    
    
    for param in net.classifier.parameters():
        param.requires_grad = True
    # net.to(device)
        
    loss_function = nn.CrossEntropyLoss()
    params_to_update = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    print(net)

    #resume train
    start_epoch = -1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    if resume:
        if os.path.isfile('checkpoint'):
            checkpoint = torch.load('checkpoint')
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']+1))
        else:
            print("=> no checkpoint found")

    best_acc = 0.0
    save_pth_path = os.path.join(output_dir, 'AlexNet.pth')
    train_steps = len(train_loader)
    val_steps = len(validate_loader)
    new_start = 0 if start_epoch==-1 else start_epoch

    for images, labels in train_loader:
        print(f"Unique labels in batch: {torch.unique(labels)}")
        break 

    for epoch in range(start_epoch + 1, new_start+epochs):
        # train
        net.train()
        train_acc = 0.0
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += torch.eq(predict_y, labels.to(device)).sum().item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        train_accurate = train_acc / train_num

        # validate
        net.eval()
        val_acc = 0.0  # accumulate accurate number / epoch
        valrunning_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                val_loss = loss_function(outputs, val_labels.to(device))
                valrunning_loss += val_loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                val_acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = val_acc / val_num
        print('[epoch %d] train_loss: %.3f train_acc: %.3f  val_loss: %.3f val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, train_accurate, valrunning_loss / val_steps, val_accurate))
        
        loss_train.append(running_loss / train_steps)
        acc_train.append(train_accurate)
        loss_val.append(valrunning_loss / val_steps)
        acc_val.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_pth_path)
        
        checkpoint = {
            'epoch': epoch,
            'model': net.state_dict(),
            'opt _, _ imizer': optimizer.state_dict(),
        }
        #torch.save(checkpoint,'checkpoint')

    matplot_loss(loss_train, loss_val, save_path = os.path.join(output_dir,'loss.jpg'))
    matplot_acc(acc_train, acc_val, save_path = os.path.join(output_dir,'acc.jpg'))
    #save
    
    record = []
    record.append(loss_train)
    record.append(loss_val)
    record.append(acc_train)
    record.append(acc_val)
    names = ['loss_train','loss_val','acc_train','acc_val']

    df = pd.DataFrame(data=record)
    csv_save_path = os.path.join(output_dir, 'traindata.csv')
    df.to_csv(csv_save_path)  

    print('Finished Training. Results are in {output_dir}')


if __name__ == '__main__':
    main()
