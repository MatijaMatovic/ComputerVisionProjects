import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import config

class ImageDataset(Dataset):
    def __init__(self, root_dir, model='srgan'):
        super(Dataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)
        self.transforms = config.sr_transforms if model == 'srgan' else config.esr_transforms

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_file, label = self.data[index]
        class_dir = os.path.join(self.root_dir, self.class_names[label])

        image = np.array(Image.open(os.path.join(class_dir, img_file)))

        image = self.transforms['both'](image=image)['image']
        high_res = self.transforms['high_res'](image=image)['image']
        low_res = self.transforms['low_res'](image=image)['image']
        return low_res, high_res
        