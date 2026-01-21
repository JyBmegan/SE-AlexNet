import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from se_model import AlexNetWithSE, save_model_structure
# from modelvgg import vgg
# from modelalex import AlexNet
# from modeldense import densenet201
# from modelmobile import mobilenet_v3_large
# from modelefficient import efficientnetv2_s



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--se_pos', type = int, help = '1-4')
    parser.add_argument('--reduction', type = int, help = '2/4/8/16/32')
    parser.add_argument('--trans', type = str, default = 'Face', help = 'Face or Object')
    parser.add_argument('--mask', type = str, default = 'Full', help = 'Full/E/N/M')
    args = parser.parse_args()

    current_se_pos = args.se_pos
    squeeze_ratio = args.reduction
    trans_con = args.trans
    mask_con = args.mask

    print (f"Prediction: Pos = {current_se_pos}, Ratio = {squeeze_ratio}, TransCondition = {trans_con}, MaskCondition = {mask_con}")

    # # If you want to train and change the condition by yourself, uncommand these lines
    # current_se_pos = 1 # Check point Same as train.py
    # squeeze_ratio = 4 # Check point
    # trans_con = "Face" # Check point
    # mask_con = "Full" # Check point

    exp_name = f"SeC{current_se_pos}_{trans_con}Based_squeeze{squeeze_ratio}"
    root_path = r'/home/zhang/share/home/scz6112/AffectNet/ConvResults'
    exp_dir = os.path.join(root_path, exp_name)
    inf_dir = os.path.join(exp_dir, 'predict')

    os.makedirs(inf_dir, exist_ok = True)

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         #transforms.Lambda(lambda x: x.repeat(3,1,1)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    a = [] 

    # load model
    model = AlexNetWithSE(num_classes=8, se_pos = current_se_pos).to(device)
    weights_path = os.path.join(exp_dir, 'AlexNet.pth')
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path), strict = False)
    model.eval()

    # print and save model
    save_model_structure(model, exp_dir, filename="current_model_arch")

    for i in range(21):
        print(i)
        img_path = f"/home/zhang/share/home/scz6112/AffectNet/test_data/{mask_con}/{mask_con}_" +str(i+1) + ".jpg"
        #img_path = "./f" + str(i+1) + ".jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # create model
        # model = AlexNetWithSE(num_classes=2).to(device)
        # model = vgg(model_name="vgg16", num_classes=2).to(device)
        # model = densenet201(num_classes=2).to(device)
        # model = mobilenet_v3_large(num_classes=2).to(device)
        # model = efficientnetv2_s(num_classes=2).to(device)

        # load model weights

        

        
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            print(predict)

        #输出excel格式（半自动）                                                                                   
        a.append(predict[1].numpy())
        
        #plt.show()
    print(a)
    df = pd.DataFrame(a)
    output_csv_name =  os.path.join(inf_dir, f'{exp_name}_{mask_con}.csv')
    df.to_csv(output_csv_name) 

if __name__ == '__main__':
    main()