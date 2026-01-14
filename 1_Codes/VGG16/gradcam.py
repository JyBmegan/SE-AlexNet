"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image, ImageDraw
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images, preprocess_image

from model import vgg


import os

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
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
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
    # def apply_mask_to_gradcam(self, gradcam, mask): 
    #     # Ensure mask is the same size as the Grad-CAM output 
    #     mask_resized = np.array(mask.resize((gradcam.shape[1], gradcam.shape[0]), Image.ANTIALIAS)) 
    #     # Normalize the Grad-CAM heatmap to 0-1 range 
    #     gradcam = np.array(gradcam) 
    #     gradcam = (gradcam - np.min(gradcam)) / (np.max(gradcam) - np.min(gradcam)) 
    #     # Apply the mask to the heatmap, scaling the heatmap where the mask is 1 
    #     gradcam = gradcam * mask_resized 
    #     # Normalize the masked heatmap back to 0-255 range for visualization 
    #     gradcam = np.uint8(gradcam * 255) 
    #     return gradcam 
    # def create_mouth_and_nose_mask(self, image_width, image_height): 
    #     mask = Image.new('L', (image_width, image_height), 0) 
    #     # 'L' for grayscale mode 
    #     draw = ImageDraw.Draw(mask) 
    #     # Define coordinates of the mouth and nose region (example coordinates) 
    #     left = int(image_width * 0.35) 
    #     right = int(image_width * 0.65) 
    #     top = int(image_height * 0.4) 
    #     bottom = int(image_height * 0.55) 
    #     # Draw a white rectangle in the region of interest (mouth and nose) 
    #     draw.rectangle([left, top, right, bottom], fill=255) 
    #     return mask 





if __name__ == '__main__':
    base_output_dir = "/media/zhang/97e9fbd4-1a76-43b2-a56c-570c3f238fa9/se-alexnet/GradCAMHeatMap/Output"                  
    exp_name = "VGG-16-ObjectBased-raw-E"                                         # check point 8
    gradcam_save_dir = os.path.join(base_output_dir, "gradcam", exp_name)
    datapoint_save_dir = os.path.join(base_output_dir, "datapoint", exp_name)

    file_prefix = exp_name.split('-')[-1]

    if not os.path.exists(gradcam_save_dir):
        os.makedirs(gradcam_save_dir)
    if not os.path.exists(datapoint_save_dir):
        os.makedirs(datapoint_save_dir)
    
    for i, filename in enumerate(os.listdir('/home/zhang/share/home/scz6112/AffectNet/ALLResultsCollection/ClassifiedWithCondition/input_images/E'), 1):  # check point 4: Full or M or N or E
    #for i in range(300):
        img_path = '/home/zhang/share/home/scz6112/AffectNet/ALLResultsCollection/ClassifiedWithCondition/input_images/E/' + str(filename)  # check point 7: Full or M or N or E
        #img_path = './input_images/vmer/A_11gr.jpg'
        file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
        # Read image
        original_image = Image.open(img_path).convert('RGB')
        # Process image
        prep_img = preprocess_image(original_image)
        # Define model

        # create model
        model = vgg(model_name="vgg16", num_classes=2)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (vgg().to(device))
        # load model weights
        weights_path = "/home/zhang/share/home/scz6112/AffectNet/ALLResultsCollection/ClassifiedWithCondition/VGG16/ObjectBased/vgg16Net.pth"  # check point 5: SE Location/ Face or Object/ squeeze
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path), strict = False)

        # pretrained_dict = torch.load(weights_path)
        # model_dict = model.state_dict()
        # filtered_dict = {k: v for k, v in pretrained_dict.items () if k in model_dict}
        # model_dict.update(filtered_dict)
        # model.load_state_dict(model_dict)

        # model.load_state_dict(torch.load(weights_path), strict = False)
        pretrained_model = model
        # print(pretrained_model)
        # file = open("./model.txt", 'w')
        # file.write(str(pretrained_model))
        # file.close()
        # Layer cam3
        layer_cam = GradCam(pretrained_model, target_layer='18')                                                # check point 6: for different SE location
        # Generate cam mask
        cam = layer_cam.generate_cam(prep_img)

        base_filename = "{}_{}".format(file_prefix, i)
        txt_path = os.path.join(datapoint_save_dir, "{}.txt".format(base_filename))
        np.savetxt(txt_path, cam)

        # Save mask
        image_path_prefix = os.path.join(gradcam_save_dir, "{}.png".format(base_filename))
        save_class_activation_images(original_image, cam, image_path_prefix,filename)
        # print('Grad cam completed')
        # print('saving path', file_name_to_export,filename)


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