import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        '''
        A pretrained InceptionNET will be used as the encoder part, as I don't have enough processing power
        '''
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        # We will fine tune the FC layer though
        self.inception.fc = nn.Linear(
            in_features=self.inception.fc.in_features, 
            out_features=embed_size
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        encoding = self.inception(images)

        for name, param in self.inception.named_parameters():
            if 'fc.weight' in name or 'fc.bias' in name:
                # always train FC layer
                param.requires_grad = True
            else:
                # train the conv part if we wish
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(encoding))
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        # TODO: Improve by adding attention
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoding, captions):
        '''
        Important note: the input for the n-th time
        step of the RNN is not the n-1-th output,
        but instead, the correct n-1th part of the caption. I.e. teacher forcing is 100% 
        '''
        embeddings = self.dropout(self.embed(captions))
        # the input encoding will serve as the first input
        embeddings = torch.cat((encoding.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    
class CaptionerNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionerNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        encoding = self.encoder(images)
        outputs = self.decoder(encoding, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        '''
        Forward function is used for training and it
        forwards the test set inputs to the decoder
        (teacher forcing). In eval mode, that is not
        desireable, so this function implements the
        actual previous predictions in each time step
        to the model
        '''
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, state = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)  # word with highest prob
                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>": break

        return [vocabulary.itos[idx] for idx in result_caption]
    

