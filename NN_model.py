# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
from DataHandlers import *

def consistency_reg(out_vid):
    loss_term = 0
    for i in range(1, out_vid.shape[1]):
        loss_term += torch.sum(torch.abs(out_vid[0, i] - out_vid[0, i-1]))
    return loss_term / out_vid.shape[1]

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1e-4):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]
        count_h = self._tensor_size(x[:, :, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, :, 1:])
        h_tv = torch.pow((x[:, :, :, 1:, :] - x[:, :, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, :, 1:] - x[:, :, :, :, :w_x-1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[-2] * t.size()[-1]

# ============================================ This is working fine ===================================================

class ConvBLSTM(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels, num_layers, device):
        super(ConvBLSTM, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.device = device
        self.forward_net = ConvLSTM(input_size, input_channels, hidden_channels, kernel_size=(5, 5), num_layers=num_layers, device=device)
        self.reverse_net = ConvLSTM(input_size, input_channels, hidden_channels, kernel_size=(5, 5), num_layers=num_layers, device=device)
        self.conv_net = nn.Sequential(nn.Conv2d(2 * self.hidden_channels, 128, kernel_size=5, padding=2),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 256, kernel_size=5, padding=2),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 64, kernel_size=5, padding=2),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 1, kernel_size=5, padding=2))

    def forward(self, xforward, xreverse, hidden_fwd=None, prop_hidden=False):
        """
        xforward, xreverse = B T C H W tensors.
        """

        hidden_bwd = None
        '''if(hidden_fwd == None):
            hidden_bwd = None
        else:
            hidden_bwd = []
            for j in range(self.num_layers):
                hidden_bwd.append([torch.zeros_like(hidden_fwd[0][0]), torch.zeros_like(hidden_fwd[0][0])])'''

        y_out_fwd, hidden_fwd = self.forward_net(xforward, hidden_fwd, prop_hidden)
        y_out_rev, _ = self.reverse_net(xreverse, hidden_bwd, prop_hidden)

        # Take only the output of the last layer
        y_out_fwd = y_out_fwd[-1]
        y_out_rev = y_out_rev[-1]

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        # reverse temporal outputs.
        y_out_rev = y_out_rev[:, reversed_idx, ...]
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)

        curr_out = []
        for j in range(ycat.shape[1]):
            curr_out.append(self.conv_net(ycat[:, j]))
        out = torch.stack(curr_out, dim=1)

        if(prop_hidden):
            return out, hidden_fwd
        else:
            return out, None

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, device):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        device: string
            Specify the device
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.device = device

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, tensor_size, device):
        height, width = tensor_size
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).to(device),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).to(device))

class ConvLSTM(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers, device='cpu',
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.device = device
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias, device=self.device))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, prop_hidden=False):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state:
            None.
        Returns
        -------
        last_state_list, layer_output
        """
        self.input_device = input_tensor.device
        # Implement stateful ConvLSTM
        if prop_hidden:
            self.return_all_layers = True
            if hidden_state == None:
                tensor_size = (input_tensor.size(3), input_tensor.size(4))
                hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)
        else:
            tensor_size = (input_tensor.size(3), input_tensor.size(4))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0), tensor_size=tensor_size)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, tensor_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, tensor_size, self.input_device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

# =====================================================================================================================
class ConvLSTM_DeepSTORM(nn.Module):
    def __init__(self, input_size, input_channels, hidden_channels, num_layers, device):
        super(ConvLSTM_DeepSTORM, self).__init__()
        self.input_size = input_size
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.device = device

        # Defining the layers
        self.dl_rnn = ConvLSTM(input_size, input_channels, hidden_channels, kernel_size=(5, 5), num_layers=num_layers, device=device)

        self.conv_net = nn.Sequential(nn.Conv2d(self.hidden_channels, 64, kernel_size=5, padding=2),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 256, kernel_size=5, padding=2),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 64, kernel_size=5, padding=2),
                                      nn.ReLU(),
                                      nn.Conv2d(64, 1, kernel_size=5, padding=2))
    def forward(self, x):
        h, _ = self.dl_rnn(x)
        curr_out = []
        for j in range(h[-1].shape[1]):
            curr_out.append(self.conv_net(h[-1][:, j]))
        out = torch.stack(curr_out, dim=1)
        return out

class RNN_DeepSTORM(nn.Module):
    def __init__(self, hidden_channels, input_range, device):
        super(RNN_DeepSTORM, self).__init__()
        self.hidden_channels = int(hidden_channels)
        self.input_range = input_range
        self.device = device

        # Defining the layers
        # Dynamics path
        self.conv_dyn = nn.Sequential(nn.Conv2d(int(2*self.input_range + 1), 256, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(64, 1, kernel_size=3, padding=1))

        # Structure path
        self.conv_dl = nn.Sequential(nn.Conv2d(1, 64, kernel_size=5, padding=2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 256, kernel_size=5, padding=2),
                                     nn.LeakyReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(256, 64, kernel_size=5, padding=2),
                                     nn.LeakyReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(64, 1, kernel_size=5, padding=2))
        # hidden layers
        self.hidden2hidden = nn.Sequential(nn.Conv2d(self.hidden_channels + 1, 128, kernel_size=3, padding=1),
                                           nn.LeakyReLU(),
                                           nn.MaxPool2d(kernel_size=2, stride=2),
                                           nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                           nn.LeakyReLU(),
                                           nn.UpsamplingBilinear2d(scale_factor=2),
                                           nn.Conv2d(128, self.hidden_channels, kernel_size=3, padding=1))

        self.out_layer = nn.Sequential(nn.Conv2d(self.hidden_channels, 256, kernel_size=5, padding=2),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(256, 128, kernel_size=5, padding=2),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(128, 1, kernel_size=5, padding=2))

    def forward(self, x, dl_in, hidden):
        dyn = self.conv_dyn(x)
        dl = self.conv_dl(dl_in)

        new_hidden = self.hidden2hidden(torch.cat([hidden, dl], dim=1))

        out = self.out_layer(new_hidden + dyn)

        return out, new_hidden

    def init_hidden(self, batch_size, hidden_dim):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.normal(0, 1/(batch_size*self.hidden_channels*hidden_dim),[batch_size, self.hidden_channels, hidden_dim, hidden_dim])
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden.to(self.device)

# RNN that gets as input the observed blinking events and a diffraction limited image on each frame
class RNNandDL(nn.Module):
    def __init__(self, hidden_channels, device):
        super(RNNandDL, self).__init__()
        self.hidden_channels = int(hidden_channels)
        self.device = device

        # Defining the layers
        # input layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        # diffraction limited layers
        self.conv1dl = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2dl = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3dl = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv4dl = nn.Conv2d(32, 1, kernel_size=5, padding=2)

        # hidden layers
        self.hidden2hidden = nn.Sequential(nn.Conv2d(self.hidden_channels + 2, self.hidden_channels, kernel_size=5, padding=2),
                                           nn.ReLU(),
                                           nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=5, padding=2))

        self.hidden2out = nn.Sequential(nn.Conv2d(self.hidden_channels, 64, kernel_size=5, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 64, kernel_size=5, padding=2),
                                        nn.ReLU(),
                                        nn.Conv2d(64, 1, kernel_size=5, padding=2))

        # support layers
        self.upsamle = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU()

    def forward(self, x, dl, hidden):
        # input to hidden state
        x1 = self.conv1(x)
        x2 = self.conv2(self.relu(x1))
        x3 = self.conv3(self.relu(x2))
        x4 = self.conv4(self.relu(x3))
        x5 = self.conv5(self.relu(self.upsamle(x4)))
        x6 = self.conv6(self.relu(self.upsamle(x5)))

        # residual connection
        x1_res = self.upsamle(self.upsamle(x))

        # dl to hidden state
        dl1 = self.conv1dl(dl)
        dl2 = self.conv2dl(self.relu(dl1))
        dl3 = self.conv3dl(self.relu(dl2))
        dl4 = self.conv4dl(self.relu(dl3))

        # combine input and dl to calculate hidden state
        new_hidden = self.hidden2hidden(torch.cat([hidden, dl4, x6 + x1_res], dim=1))


        # hidden to output
        out = self.hidden2out(new_hidden)

        return out, new_hidden

    def init_hidden(self, batch_size, hidden_dim):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.normal(0, 1/(batch_size*self.hidden_channels*hidden_dim),[batch_size, self.hidden_channels, hidden_dim, hidden_dim])
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden.to(self.device)

# RNN that gets as input the observed blinking events and a diffraction limited image on each frame
class RNN_dual_path(nn.Module):
    def __init__(self, hidden_channels, device):
        super(RNN_dual_path, self).__init__()
        self.hidden_channels = int(hidden_channels)
        self.device = device

        # Defining the layers
        # Dynamics path
        self.conv_dyn = nn.Sequential(nn.Conv2d(11, 256, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.Conv2d(512, 256, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.UpsamplingBilinear2d(scale_factor=2),
                                      nn.Conv2d(256, 64, kernel_size=3, padding=1),
                                      nn.LeakyReLU(),
                                      nn.UpsamplingBilinear2d(scale_factor=2),
                                      nn.Conv2d(64, 1, kernel_size=3, padding=1))

        # Structure path
        self.conv_dl = nn.Sequential(nn.Conv2d(1, 64, kernel_size=5, padding=2),
                                     nn.LeakyReLU(),
                                     nn.Conv2d(64, 256, kernel_size=5, padding=2),
                                     nn.LeakyReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(256, 64, kernel_size=5, padding=2),
                                     nn.LeakyReLU(),
                                     nn.UpsamplingBilinear2d(scale_factor=2),
                                     nn.Conv2d(64, 1, kernel_size=5, padding=2))
        # hidden layers
        self.hidden2hidden = nn.Sequential(nn.Conv2d(self.hidden_channels + 1, 128, kernel_size=3, padding=1),
                                           nn.LeakyReLU(),
                                           nn.MaxPool2d(kernel_size=2, stride=2),
                                           nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(256, 128, kernel_size=3, padding=1),
                                           nn.LeakyReLU(),
                                           nn.UpsamplingBilinear2d(scale_factor=2),
                                           nn.Conv2d(128, self.hidden_channels, kernel_size=3, padding=1))

        self.out_layer = nn.Sequential(nn.Conv2d(self.hidden_channels, 256, kernel_size=5, padding=2),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(256, 128, kernel_size=5, padding=2),
                                       nn.LeakyReLU(),
                                       nn.Conv2d(128, 1, kernel_size=5, padding=2))


    def forward(self, x, dl_in, hidden):
        dyn = self.conv_dyn(x)
        dl = self.conv_dl(dl_in)

        new_hidden = self.hidden2hidden(torch.cat([hidden, dl], dim=1))

        out = self.out_layer(new_hidden + dyn)

        return out, new_hidden

    def init_hidden(self, batch_size, hidden_dim):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.normal(0, 1/(batch_size*self.hidden_channels*hidden_dim),[batch_size, self.hidden_channels, hidden_dim, hidden_dim])
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        return hidden.to(self.device)
