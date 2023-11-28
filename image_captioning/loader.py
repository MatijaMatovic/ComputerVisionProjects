# text -> numerical values
# 1. Create a vocabulary to map each word to an index
# 2. Set up a PyTorch dataset to load the data
# 3. Set up padding of every batch (all examples should be of same sequence length and then, set up data loader)

import os
import torch
import pandas as pd
import spacy # for tokenization

from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image

spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary:
    def __init__(self, freq_threshold) -> None:
        '''
        Args:
            - freq_treshold : if a word is not repeated a certain amount of times, defined by this parameter, do not include it in the vocab
        '''

        self.itos = {
            0: '<PAD>',
            1: '<SOS>',
            2: '<EOS>',
            3: '<UNK>',  # map the words below the threshold to this token
        }

        self.stoi = { value: key for key, value in self.itos.items() }

        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self, sentence_list):
        frequencies = defaultdict
        index = 4 # because we already have 0-3 in our initial itos
        for sentence in sentence_list:
            for word in Vocabulary.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = index
                    self.itos[index] = word
                    index += 1

    def numerize(self, text):
        tokenized_text = Vocabulary.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenized_text
        ]
            


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transforms=None, freq_treshold=5) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transforms = transforms
        
        self.img_paths = self.df['image']
        self.captions = self.df['caption']

        self.vocab = Vocabulary(freq_treshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.captions[index]
        img_path = self.img_paths[index]

        img = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        num_caption = [self.vocab.stoi['<SOS>']]
        num_caption += self.vocab.numerize(caption)
        num_caption.append(self.vocab.stoi['<EOS>'])

        return img, torch.tensor(num_caption)
    

class Collate:
    '''
    It is unnecessary to pad every sentence to the length of the longest sentence in the dataset. 
    It is sufficient to only pad every sentence to the length 
    of the longest sentence in the current batch.
    This class will be passed to the data loader, and will work on every batch
    '''
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)

        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    
def get_loader(
        root_dir, captions_file,
        transforms,
        batch_size=32, shuffle=True,
        num_workers=8, pin_memory=True
):
    dataset = FlickrDataset(root_dir, captions_file,         transforms)

    pad_idx = dataset.vocab.stoi['<PAD>']

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=Collate(pad_idx)
    )

    return loader

