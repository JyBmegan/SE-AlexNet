from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
class affectnet(Dataset):
    def __init__(self, dataset_dir, csv_path):
        self.dataset_dir = dataset_dir
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path,encoding='utf-8')
        self.transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
                                    ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        x_train = self.transform(Image.open(self.df.file_path[index]))
        # print(self.df.file_path[index])
        y_train = self.df.expression[index]
    # def __getitem__(self, index):  
    #     img_path = os.path.join(self.dataset_dir, self.df.file_path[index])  
    #     x_train = self.transform(Image.open(img_path))  
    #     y_train = self.df.valence_label[index]
    #     #x_train = self.transform(Image.open(self.dataset_dir + self.df.file_path[index]))  # Ensure the file path is correct  
    #     # Assuming the correct column name for labels is 'label' (replace 'label' with the actual column name)  
    #     #y_train = self.df.label[index]  # Replace 'label' with the correct column name  
  
        return x_train, y_train

        #return x_train, y_train  
