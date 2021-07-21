import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DatasetGenerator(Dataset):

  def __init__(self,path,image_size):
    
    self.X = glob.glob(path + '*')
    self.transform = transforms.Compose([
                     transforms.ToPILImage(),
                     transforms.Resize((image_size,image_size)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
  
  
  def __getitem__(self,idx):
    
    img = plt.imread(self.X[idx])
    img = self.transform(img)
    return img

  def __len__(self):
    
    return len(self.X)

	
