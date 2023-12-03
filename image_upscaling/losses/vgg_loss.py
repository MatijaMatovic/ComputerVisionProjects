import torch.nn as nn
from torchvision.models import vgg19

# 5th conv layer before maxpooling but after activation

'''
This loss takes the activation of a hidden layer somewhere deep in the VGG CNN both for the target, and for the input.
If our GAN has preserved the important features of the image,
these activations will be similar, which is calculated by MSE
This is better than calculating MSE on the picture itself,
because this doesn't account for the properties/features of the image, as using layers does
'''
class VGGLoss(nn.Module):
    def __init__(self, model='srgan') -> None:
        super().__init__()
        extraction_point = 35 if model == 'esrgan' else 36
        self.vgg = vgg19(pretrained=True).features[:extraction_point].eval()  # to(device)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False


    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)