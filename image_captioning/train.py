import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from loader import get_loader
from model import CaptionerNN

def train():
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),  # input size for InceptionNET
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=transform,
        num_worksers = 2
    )

    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 300

    writer = SummaryWriter('runs/flickr')

    model = CaptionerNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi['<PAD>'])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    step = 0

    for epoch in range(num_epochs):
       for idx, (imgs, captions) in enumerate(train_loader):
           imgs = imgs.to(device)
           captions = captions.to(device)

           outputs = model(imgs, captions[:-1]) # we want the model to predict the <END> 

           # (seq_len, N, vocab_size) -> (seq_len, N)
           # we don't want to predict the prob for each word, but for all the words in the sentence together
           loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

           writer.add_scalar("Training loss", loss.item(), global_step=step)

           step += 1

           optimizer.zero_grad()
           loss.backward(loss)
           optimizer.step()


if __name__ == '__main__':
    train()