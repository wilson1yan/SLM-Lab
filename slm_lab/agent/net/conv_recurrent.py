from slm_lab.agent.net import net_util
from slm_lab.agent.net.base import Net
from slm_lab.lib import logger, util
import numpy as np
import pydash as ps
import torch
import torch.nn as nn

logger = logger.get_logger(__name__)


class ConvRecurrentNet(Net, nn.Module):
    '''
    Class for generating arbitrary sized recurrent neural networks which take a sequence of states as input.

    Assumes that a single input example is organized into a 3D tensor
    batch_size x seq_len x state_dim
    The entire model consists of three parts:
        1. self.fc_model (state processing)
        2. self.rnn_model
        3. self.model_tails

    e.g. net_spec
    "net": {
        "type": "ConvRecurrentNet",
        "shared": true,
        "conv_hid_layers": [
            [32, 8, 4, 0, 1],
            [64, 4, 2, 0, 1],
            [64, 3, 1, 0, 1]
        ],
        "hid_conv_activation": "relu",
        "batch_norm": false,
        "cell_type": "LSTM",
        "fc_hid_layers": [],
        "hid_recurrent_activation": "relu",
        "rnn_hidden_size": 32,
        "rnn_num_layers": 1,
        "bidirectional": false,
        "seq_len": 4,
        "init_fn": "xavier_uniform_",
        "clip_grad_val": 1.0,
        "loss_spec": {
          "name": "MSELoss"
        },
        "optim_spec": {
          "name": "Adam",
          "lr": 0.01
        },
        "lr_scheduler_spec": {
            "name": "StepLR",
            "step_size": 30,
            "gamma": 0.1
        },
        "update_type": "replace",
        "update_frequency": 10000,
        "polyak_coef": 0.9,
        "gpu": true
    }
    '''

    def __init__(self, net_spec, in_dim, out_dim):
        '''
        net_spec:
        cell_type: any of RNN, LSTM, GRU
        fc_hid_layers: list of fc layers preceeding the RNN layers
        hid_layers_activation: activation function for the fc hidden layers
        rnn_hidden_size: rnn hidden_size
        rnn_num_layers: number of recurrent layers
        bidirectional: if RNN should be bidirectional
        seq_len: length of the history of being passed to the net
        init_fn: weight initialization function
        clip_grad_val: clip gradient norm if value is not None
        loss_spec: measure of error between model predictions and correct outputs
        optim_spec: parameters for initializing the optimizer
        lr_scheduler_spec: Pytorch optim.lr_scheduler
        update_type: method to update network weights: 'replace' or 'polyak'
        update_frequency: how many total timesteps per update
        polyak_coef: ratio of polyak weight update
        gpu: whether to train using a GPU. Note this will only work if a GPU is available, othewise setting gpu=True does nothing
        '''
        nn.Module.__init__(self)
        super(ConvRecurrentNet, self).__init__(net_spec, in_dim, out_dim)
        # set default
        util.set_attr(self, dict(
            cell_type='GRU',
            rnn_num_layers=1,
            bidirectional=False,
            init_fn=None,
            clip_grad_val=None,
            loss_spec={'name': 'MSELoss'},
            optim_spec={'name': 'Adam'},
            lr_scheduler_spec=None,
            update_type='replace',
            update_frequency=1,
            polyak_coef=0.0,
            gpu=False,
        ))
        util.set_attr(self, self.net_spec, [
            'conv_hid_layers',
            'hid_layers_activation',
            'batch_norm',
            'cell_type',
            'fc_hid_layers',
            'rnn_hidden_size',
            'rnn_num_layers',
            'bidirectional',
            'seq_len',
            'init_fn',
            'clip_grad_val',
            'loss_spec',
            'optim_spec',
            'lr_scheduler_spec',
            'update_type',
            'update_frequency',
            'polyak_coef',
            'gpu',
        ])

        # conv layer
        self.conv_model = self.build_conv_layers(self.conv_hid_layers)
        self.conv_out_dim = self.get_conv_output_size()

        self.rnn_input_dim = self.conv_out_dim

        # RNN model
        self.rnn_model = getattr(nn, self.cell_type)(
            input_size=self.rnn_input_dim,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_num_layers,
            batch_first=True, bidirectional=self.bidirectional)

        # tails. avoid list for single-tail for compute speed
        if ps.is_integer(self.out_dim):
            self.model_tail = nn.Linear(self.rnn_hidden_size, self.out_dim)
        else:
            self.model_tails = nn.ModuleList([nn.Linear(self.rnn_hidden_size, out_d) for out_d in self.out_dim])

        net_util.init_layers(self, self.init_fn)
        for module in self.modules():
            module.to(self.device)
        self.loss_fn = net_util.get_loss_fn(self, self.loss_spec)
        self.optim = net_util.get_optim(self, self.optim_spec)
        self.lr_scheduler = net_util.get_lr_scheduler(self, self.lr_scheduler_spec)

    def __str__(self):
        return super(ConvRecurrentNet, self).__str__() + f'\noptim: {self.optim}'


    def get_conv_output_size(self):
        '''Helper function to calculate the size of the flattened features after the final convolutional layer'''
        with torch.no_grad():
            x = torch.ones(1, 1,  *self.in_dim[1:])
            x = self.conv_model(x)
            return x.numel()

    def build_conv_layers(self, conv_hid_layers):
        '''
        Builds all of the convolutional layers in the network and store in a Sequential model
        '''
        conv_layers = []
        in_d = 1  # input channel
        for i, hid_layer in enumerate(conv_hid_layers):
            hid_layer = [tuple(e) if ps.is_list(e) else e for e in hid_layer]  # guard list-to-tuple
            # hid_layer = out_d, kernel, stride, padding, dilation
            conv_layers.append(nn.Conv2d(in_d, *hid_layer))
            conv_layers.append(net_util.get_activation_fn(self.hid_layers_activation))
            # Don't include batch norm in the first layer
            if self.batch_norm and i != 0:
                conv_layers.append(nn.BatchNorm2d(in_d))
            in_d = hid_layer[0]  # update to out_d
        conv_model = nn.Sequential(*conv_layers)
        return conv_model

    def forward(self, x):
        '''The feedforward step. Input is batch_size x seq_len x state_dim'''
        print('data', x.size())
        # Unstack input to (batch_size x seq_len) x state_dim in order to transform all state inputs
        batch_size = x.size(0)
        x = x.view(-1, 1, *self.in_dim[1:])
        x = self.conv_model(x)
        x = x.view(batch_size * self.seq_len, -1)

        # Restack to batch_size x seq_len x rnn_input_dim
        x = x.view(-1, self.seq_len, self.rnn_input_dim)
        if self.cell_type == 'LSTM':
            _output, (h_n, c_n) = self.rnn_model(x)
        else:
            _output, h_n = self.rnn_model(x)
        hid_x = h_n[-1]  # get final time-layer
        # return tensor if single tail, else list of tail tensors
        if hasattr(self, 'model_tails'):
            outs = []
            for model_tail in self.model_tails:
                outs.append(model_tail(hid_x))
            return outs
        else:
            return self.model_tail(hid_x)

    def training_step(self, x=None, y=None, loss=None, retain_graph=False, lr_clock=None):
        '''Takes a single training step: one forward and one backwards pass'''
        if hasattr(self, 'model_tails') and x is not None:
            raise ValueError('Loss computation from x,y not supported for multitails')
        self.lr_scheduler.step(epoch=ps.get(lr_clock, 'total_t'))
        self.train()
        self.optim.zero_grad()
        if loss is None:
            out = self(x)
            loss = self.loss_fn(out, y)
        assert not torch.isnan(loss).any(), loss
        if net_util.to_assert_trained():
            assert_trained = net_util.gen_assert_trained(self)
        loss.backward(retain_graph=retain_graph)
        if self.clip_grad_val is not None:
            nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_val)
        self.optim.step()
        if net_util.to_assert_trained():
            assert_trained(self, loss)
            self.store_grad_norms()
        logger.debug(f'Net training_step loss: {loss}')
        return loss

    def wrap_eval(self, x):
        '''
        Completes one feedforward step, ensuring net is set to evaluation model returns: network output given input x
        '''
        self.eval()
        return self(x)
