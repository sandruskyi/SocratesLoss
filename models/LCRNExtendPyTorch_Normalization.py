import torch
from torch import nn
"""
User: 
"""
__all__ = [
    'LCRNExtendPyTorch_Normalization', 'lcrn_normalization', 'lcrn_normalization_new', 'TorchTimeDistributed', 'L1'
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class L1(torch.nn.Module):
    """
    source : https://stackoverflow.com/questions/42704283/l1-l2-regularization-in-pytorch

    Class for L1 regularisation method
    """
    def __init__(self, module, weight_decay):
        super().__init__()
        self.module = module
        self.weight_decay = weight_decay

        # Backward hook is registered on the specified module
        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    # Not dependent on backprop incoming values, placeholder
    def _weight_decay_hook(self, *_):
        for param in self.module.parameters():
            # If there is no gradient or it was zeroed out
            # Zeroed out using optimizer.zero_grad() usually
            # Turn on if needed with grad accumulation/more safer way
            # if param.grad is None or torch.all(param.grad == 0.0):

            # Apply regularization on it
            param.grad = self.regularize(param)

    def regularize(self, parameter):
        # L1 regularization formula
        return self.weight_decay * torch.sign(parameter.data)

    def forward(self, *args, **kwargs):
        # Simply forward and args and kwargs to module
        return self.module(*args, **kwargs)

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


class LCRNExtendPyTorch_Normalization(nn.Module):
    name = "LCRNExtendPyTorch_Normalization"

    def __init__(self, input_shape=(45, 3, 24, 24), classes=2, type="old", **kwargs):
        super(LCRNExtendPyTorch_Normalization, self).__init__()
        dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else 0.2
        dropout_1 = kwargs['dropout_1'] if 'dropout' in kwargs.keys() else 0.0
        dropout_2 = kwargs['dropout_2'] if 'dropout' in kwargs.keys() else 0.2
        l1_reg = kwargs['l1_reg'] if 'l1_reg' in kwargs.keys() else 0.0
        lstm_dim = kwargs['lstm_dim'] if 'lstm_dim' in kwargs.keys() else 128
        dense_dim = kwargs['dense_dim'] if 'dense_dim' in kwargs.keys() else 256
        conv_dim = kwargs['conv_dim'] if 'conv_dim' in kwargs.keys() else 32
        activation = kwargs['activation'] if 'activation' in kwargs.keys() else "selu"
        gru = kwargs['gru'] if 'gru' in kwargs.keys() else True
        self.classes = classes
        if type=="old":
            self.architecture = nn.Sequential(
                TorchTimeDistributed(nn.Conv2d(input_shape[-3], conv_dim, kernel_size=3, padding="same")),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                TorchTimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
                TorchTimeDistributed(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, padding="same")),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                TorchTimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
                TorchTimeDistributed(nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, padding="same")),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                nn.Flatten(start_dim=2),
                nn.Linear(conv_dim * 4 * ((input_shape[2] // 4) ** 2), dense_dim),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(dense_dim, dense_dim // 2),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                nn.Dropout(p=dropout)
            )
        else:
            self.architecture = nn.Sequential(
                TorchTimeDistributed(nn.Conv2d(input_shape[-3], conv_dim, kernel_size=9, padding="same")),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                # nn.Dropout(p=0.05),  # Added dropout

                TorchTimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
                TorchTimeDistributed(nn.Conv2d(conv_dim, conv_dim * 2, kernel_size=9, padding="same")),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                nn.Dropout(p=dropout_1),  # Added dropout

                TorchTimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
                TorchTimeDistributed(nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=9, padding="same")),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                nn.Dropout(p=dropout_1),  # Added dropout

                nn.Flatten(start_dim=2),
                nn.LazyLinear(dense_dim),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                nn.Dropout(p=dropout_2),  # Increased dropout

                nn.Linear(dense_dim, dense_dim // 2),
                nn.SELU() if activation == "selu" else nn.ReLU(),
                nn.Dropout(p=dropout_2)  # Increased dropout
            )
            self.architecture = L1(self.architecture, weight_decay=l1_reg)  # Adding L1 regularisation method

        # Recurrent layer
        self.recurrent = nn.GRU(dense_dim // 2, lstm_dim, batch_first=True) if gru else \
            nn.LSTM(dense_dim // 2, lstm_dim, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(lstm_dim, classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        conv_out = self.architecture(x)
        r_out, r_hidden = self.recurrent(conv_out)
        sig_out = self.sigmoid(r_out[:, -1, :])
        return self.linear(sig_out)


    def predict(self, x, **kwargs):
        with torch.no_grad():
            if type(x) is not torch.Tensor:
                x_t = torch.Tensor(x).to(self.device)
            else:
                x_t = x.to(self.device)
            out = self.forward(x_t)
            out = nn.Sigmoid()(out) if self.classes == 1 else nn.Softmax(dim=1)(out) # We want to use softmax all the time, if not change it self.classes==2
        return out.cpu()

def lcrn_normalization(dataset_name, **kwargs):
    name = "LCRNExtendPyTorch_Normalization"
    type = "old"
    model = LCRNExtendPyTorch_Normalization(type=type, **kwargs)
    return model
def lcrn_normalization_new(dataset_name, **kwargs):
    name = "LCRNExtendPyTorch_Normalization"
    type = "new"
    model = LCRNExtendPyTorch_Normalization(type=type, **kwargs)
    return model