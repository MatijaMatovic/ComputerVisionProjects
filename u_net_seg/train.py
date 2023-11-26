import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from tqdm import tqdm
from u_net import UNET
from albumentations.pytorch import ToTensorV2

from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy
)

LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/train_images/'
TRAIN_MASK_DIR = 'data/train_masks/'
VAL_IMG_DIR = 'data/val_images/'
VAL_MASK_DIR = 'data/val_masks/'

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    train_transforms = A.compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
            max_pixel_value=255.
        ),
        ToTensorV2(),
    ])

    val_transforms = A.compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0., 0., 0.],
            std=[1., 1., 1.],
            max_pixel_value=255.
        ),
        ToTensorV2()
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()  # adds sigmoid. if there was more than 1 class, it would be just crossentropy instead of binary c-e

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transforms,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load('u_ckp.pth.tar'))

    scaler = torch.cuda.amp.GradScaler()
    for _ in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        check_accuracy(val_loader, model, DEVICE)



if __name__ == '__main__':
    main()