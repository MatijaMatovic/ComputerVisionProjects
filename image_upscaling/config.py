import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

SRGAN_CONFIG = {
    'load_model': False,
    'save_model': True,
    'checkpoint_gen': 'sr_gen.pth.tar',
    'checkpoint_disc': 'sr_disc.pth.tar',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'learning_rate': 1e-4,
    'num_epochs': 100,
    'num_pretrain_epochs': 10,
    'batch_size': 16,
    'num_workers': 4,
    'high_res': 96,
    'low_res_scale': 4,
    'color_channels': 3
}

ESRGAN_CONFIG = {
    'load_model': False,
    'save_model': True,
    'checkpoint_gen': 'esr_gen.pth.tar',
    'checkpoint_disc': 'esr_disc.pth.tar',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'learning_rate': 1e-4,
    'num_epochs': 1000,
    'batch_size': 16,
    'lambda_gp': 10,
    'num_workers': 4,
    'high_res': 128,
    'low_res_scale': 4,
    'color_channels': 3
}

sr_high_res_transform = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])

sr_low_res_dim = SRGAN_CONFIG['high_res'] // SRGAN_CONFIG['low_res_scale']
sr_low_res_transform = A.Compose([
    A.Resize(width=sr_low_res_dim, height=sr_low_res_dim, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
])

sr_both_transform = A.Compose([
    A.RandomCrop(width=SRGAN_CONFIG['high_res'], height=SRGAN_CONFIG['high_res']),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

sr_test_transform = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
])


sr_transforms = {
    'high_res': sr_high_res_transform,
    'low_res': sr_low_res_transform,
    'both': sr_both_transform,
    'test': sr_test_transform
}


esr_high_res_transform = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
])

esr_low_res_dim = ESRGAN_CONFIG['high_res'] // ESRGAN_CONFIG['low_res_scale']
esr_low_res_transform = A.Compose([
    A.Resize(width=esr_low_res_dim, height=esr_low_res_dim, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
])

esr_both_transform = A.Compose([
    A.RandomCrop(width=ESRGAN_CONFIG['high_res'], height=ESRGAN_CONFIG['high_res']),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])

esr_test_transform = A.Compose([
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2()
])



esr_transforms = {
    'high_res': esr_high_res_transform,
    'low_res': esr_low_res_transform,
    'both': esr_both_transform,
    'test': esr_test_transform
}