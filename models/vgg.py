import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG', 'vgg16', 'vgg16_bn'
]

class VGG(nn.Module):
    name = "vgg16"
    def __init__(self, features, dataset_name = "cacophony", num_classes=2, input_size = -1):
        super(VGG, self).__init__()
        self.dataset = dataset_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.features = features
        self.classes = num_classes

        if input_size == 32:
            self.classifier = nn.Sequential(nn.Linear(512,512), nn.ReLU(inplace=True), \
                                        nn.BatchNorm1d(512),nn.Dropout2d(0.5),nn.Linear(512, num_classes))
        elif input_size == 64:
            self.classifier = nn.Sequential(nn.Linear(2048,512), nn.ReLU(inplace=True), \
                                        nn.BatchNorm1d(512),nn.Dropout2d(0.5),nn.Linear(512, num_classes))
        elif input_size == 224:
            self.classifier = nn.Sequential(nn.Linear(512*7*7,4096), nn.ReLU(inplace=True), \
                                        nn.BatchNorm1d(4096),nn.Dropout2d(0.5),nn.Linear(4096, num_classes))
        if dataset_name == "cacophony":
            input_shape = (45, 3, 24, 24)
            conv_dim = 32
            #self.classifier = nn.Sequential(nn.Linear(3072, 512), nn.ReLU(inplace=True), \
            #                                nn.BatchNorm1d(512), nn.Dropout2d(0.5), nn.Linear(512, num_classes))
            self.classifier = nn.Sequential(nn.Linear(conv_dim * 4 * ((input_shape[2] // 4) ** 2), 256), nn.ReLU(inplace=True), \
                                            nn.Dropout2d(0.5), nn.Linear(256, num_classes))

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        if self.dataset != "cacophony":
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        if self.dataset != "cacophony":
            return x
        else:
            return x[:, -1, :]

    def predict(self, x, **kwargs):
        with torch.no_grad():
            if type(x) is not torch.Tensor:
                x_t = torch.Tensor(x).to(self.device)
            else:
                x_t = x.to(self.device)
            out = self.forward(x_t)
            out = nn.Sigmoid()(out) if self.classes == 1 else nn.Softmax(dim=-1)(out)
        return out.cpu()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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


def make_layers_cacophony(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [TorchTimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2))]
        elif type(v)==int:
            conv2d = TorchTimeDistributed(nn.Conv2d(in_channels, v, kernel_size=3, padding="same"))
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), TorchTimeDistributed(nn.BatchNorm2d(v))]

                # the order is modified to match the model of the baseline that we compare to
            else:
                #layers += [conv2d, nn.ReLU(inplace=True)]
                layers += [conv2d, nn.ReLU(inplace=True)]

            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout2d(v)]
    layers += [nn.Flatten(start_dim=2)]
    return nn.Sequential(*layers)

#cfg_caco = {
#    'D': [64,0.3, 64, 'M', 128,0.4, 128, 'M', 256,0.4, 256,0.4, 256, 'M', 512,0.4, 512,0.4, 512, 'M', 0.5]
#}
"""
cfg_caco = {
    'D': [32, 0.3, 'M', 64, 0.4, 'M', 128, 0.4, 'M', 0.5]
}
"""
cfg_caco = {
    'D': [32, 0.3, 'M', 64,  0.4, 'M', 128, 0.4]
}

cfg = {
    'D': [64, 0.3, 64, 'M', 128,0.4, 128, 'M', 256,0.4, 256,0.4, 256, 'M', 512,0.4, 512,0.4, 512, 'M', 512,0.4, 512,0.4, 512, 'M',0.5]
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif type(v)==int:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                # the order is modified to match the model of the baseline that we compare to
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        elif type(v)==float:
            layers += [nn.Dropout2d(v)]
    return nn.Sequential(*layers)





def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(dataset_name, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    if dataset_name == "cacophony":
        model = VGG(make_layers_cacophony(cfg_caco['D'], batch_norm=True), dataset_name, **kwargs)
    else:
        model = VGG(make_layers(cfg['D'], batch_norm=True), dataset_name, **kwargs)
    return model

