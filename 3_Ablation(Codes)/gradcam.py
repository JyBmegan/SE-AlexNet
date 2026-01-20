"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os

from misc_functions import save_class_activation_images, preprocess_image

from se_model import AlexNetWithSE




class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if str(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        #self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        min_val = np.min(cam)
        if max_val - min_val > 0:
            cam = (cam - min_val) / (max_val - min_val)  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        #from scipy.ndimage.interpolation import zoom
        #cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam
    
def find_last_conv_layer(model):
    last_conv_idx = -1
    for name, module in model.features.named_children():
        if isinstance(module, nn.Conv2d):
            last_conv_idx = name
    return last_conv_idx

if __name__ == '__main__':
    
    current_se_pos = 1 # Check point Same as train.py
    squeeze_ratio = 32 # Check point
    trans_con = "Face" # Check point
    mask_con = "Full" # Check point
    exp_name = f"SeC{current_se_pos}_{trans_con}Based_squeeze{squeeze_ratio}"
    root_path = r'/home/zhang/share/home/scz6112/AffectNet/ConvResults'
    exp_dir = os.path.join(root_path, exp_name)
    weights_path = os.path.join(exp_dir, 'AlexNet.pth')
    gradcam_dir = os.path.join(exp_dir, 'gradcam')
    datapoint_dir = os.path.join(exp_dir, 'datapoint')

    if not os.path.exists(gradcam_dir):
        os.makedirs(gradcam_dir)
    if not os.path.exists(datapoint_dir):
        os.makedirs(datapoint_dir)

    model = AlexNetWithSE(num_classes = 8, se_pos = current_se_pos)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    else: 
        print(f"Error: weights not found at {weights_path}")
        exit()
    
    target_layer_id = find_last_conv_layer(model)
    grad_cam = GradCam(model, target_layer=target_layer_id)
    
    for i, filename in enumerate(sorted(os.listdir(f'/home/zhang/share/home/scz6112/AffectNet/test_data/{mask_con}')), 1): 
    #for i in range(300):
        img_path = f"/home/zhang/share/home/scz6112/AffectNet/test_data/{mask_con}/" + str(filename) 
        #img_path = './input_images/vmer/A_11gr.jpg'
        file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
        # Read image
        original_image = Image.open(img_path).convert('RGB')
        # Process image
        prep_img = preprocess_image(original_image)
        # Define model

        # create model
        # model = AlexNetWithSE(num_classes=2)

        print (AlexNetWithSE().to(device))
        # load model weights
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path))
        pretrained_model = model
        # print(pretrained_model)
        # file = open("./model.txt", 'w')
        # file.write(str(pretrained_model))
        # file.close()
        # Layer cam3
        # layer_cam = GradCam(pretrained_model, target_layer='8')                    # check point 6: for different SE location
        # Generate cam mask
        cam = grad_cam.generate_cam(prep_img)
        # Save mask
        csv_path = os.path.join(datapoint_dir, f"{mask_con}_{i}.csv")
        np.savetxt(csv_path, cam)
        img_path = os.path.join(gradcam_dir, f"{mask_con}_{i}.png")
        save_class_activation_images(original_image, cam, img_path)
        print('Grad cam completed')


    # if __name__ == '__main__':
    #     # Get params
    #     for i in range(31):
    #         target_example = 0  # Snake
    #         (original_image, prep_img, file_name_to_export, pretrained_model) =\
    #             get_example_params(target_example)
    #         # Grad cam
    #         grad_cam = GradCam(pretrained_model, target_layer=i)
    #         # Generate cam mask
    #         cam = grad_cam.generate_cam(prep_img)
    #         # Save mask
    #         save_class_activation_images(original_image, cam, file_name_to_export,i)
    #         print('Grad cam completed')
