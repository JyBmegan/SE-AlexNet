import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import pandas as pd

from se_model import AlexNetWithSE
# from modelvgg import vgg
# from modelalex import AlexNet
# from modeldense import densenet201
# from modelmobile import mobilenet_v3_large
# from modelefficient import efficientnetv2_s



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         #transforms.Lambda(lambda x: x.repeat(3,1,1)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    a = [] 
    # load image
    for i in range(21):
        print(i)
        img_path = "/media/zhang/97e9fbd4-1a76-43b2-a56c-570c3f238fa9/yfLi/project-paper3/test_data/maskface-affectnet/original/bm_ave_" +str(i+1) + ".jpg"
        #img_path = "./f" + str(i+1) + ".jpg"
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # create model
        model = AlexNetWithSE(num_classes=2).to(device)
        # model = vgg(model_name="vgg16", num_classes=2).to(device)
        # model = densenet201(num_classes=2).to(device)
        # model = mobilenet_v3_large(num_classes=2).to(device)
        # model = efficientnetv2_s(num_classes=2).to(device)

        # load model weights
        weights_path = "/home/zhang/share/home/scz6112/AffectNet/ALLResultsCollection/ClassifiedWithCondition/SELocate-1/FaceBased/squeeze-32/AlexNet.pth"  # check point 5: the file name; If wrong, see point 4 in train.py
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path), strict = False)

        model.eval()
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
    df.to_csv('/home/zhang/share/home/scz6112/AffectNet/ALLResultsCollection/ClassifiedWithCondition/SELocate-1/FaceBased/squeeze-32/f-1-face-32.csv') # check point 6: the file name;f means full face compared with mask

if __name__ == '__main__':
    main()