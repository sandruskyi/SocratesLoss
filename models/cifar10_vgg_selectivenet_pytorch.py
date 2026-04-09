import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    'cifar10vgg_torch', 'cifar10vgg_pytorch',  'TorchTimeDistributed'
]

class TorchTimeDistributed(nn.Module):
    def __init__(self, module):
        super(TorchTimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.shape) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(x.shape[0] * x.shape[1], *x.shape[2:])  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(*x.shape[0:2], *y.shape[-3:])  # (samples, timesteps, output_size)
        return y

class cifar10vgg_torch(nn.Module):
    name = "cifar10vgg_torch"
    def __init__(self, features,  input_size = -1, classes= 10, **kwargs):
        super(cifar10vgg_torch, self).__init__()
        self.classes = classes
        self.features = features
        self.body_classifier = nn.Sequential(nn.Linear(512,512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
                                             nn.Dropout2d(0.5))

        # Classification head (f):
        self.classification_head = nn.Linear(512, classes) # Activation softmax or sigmoid

        # Selection Head (g):
        self.selection_head = nn.Sequential(nn.Linear(512,512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
                                             nn.Linear(512,1)) # Activation: Sigmoid

        # Auxiliary head (h)
        self.auxiliary_head = nn.Linear(512, classes) # Activation softmax or sigmoid. Same as f


        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, x):
        # MAIN BODY BLOCK: FEATURES + CLASSIFIER
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten all dimensions except batch
        x = self.body_classifier(x)

        # PREDICTION:
        classification_output = self.classification_head(x)

        # SELECTION:
        selection_output = self.selection_head(x)
        # AUXILIARY:
        auxiliary_output = self.auxiliary_head(x)

        return torch.cat((classification_output, selection_output), dim = 1), auxiliary_output


def make_layers_features(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif type(v)==int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding="same")
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout2d(v)]
    return nn.Sequential(*layers)

cfg = {
    'D': [64,0.3, 64, 'M', 128,0.4, 128, 'M', 256,0.4, 256,0.4, 256, 'M', 512,0.4, 512,0.4, 512, 'M', 512,0.4, 512,0.4, 512, 'M',0.5]
}

def cifar10vgg_pytorch(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = cifar10vgg_torch(make_layers_features(cfg['D'], batch_norm=True), **kwargs)
    return model