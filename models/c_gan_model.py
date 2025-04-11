### c_gan_model.py
import torch
from .basic_model import BasicModel
from collections import OrderedDict
from . import networks
import torch.nn as nn
import numpy as np

class CGanModel(BasicModel):
    @staticmethod
    def modify_commandline_parameters(parser, is_train=True):
        parser.set_defaults(netG='fcg_sep', netD='fcd_sep')
        parser.add_argument('--latent_dim', type=int, default=256, help='the dimensionality of the latent space')
        if is_train:
            parser.add_argument('--lambda_dist', type=float, default=100.0, help='weight for the dist loss')
        return parser

    def __init__(self, param):
        BasicModel.__init__(self, param)
        self.model_names = ['G']
        self.netG = networks.define_G(param.input_chan_num, param.output_chan_num, param.netG, param.A_dim, param.B_dim,
                                      param.gen_filter_num, param.conv_k_size, param.norm_type, param.init_type,
                                      param.init_gain, self.gpu_ids, param.leaky_slope, param.dropout_p, param.latent_dim)
        if self.param.zo_norm:
            self.sigmoid = nn.Sigmoid()

    def set_input(self, input_dict):
        if self.param.ch_separate:
            self.real_B_tensor = [input_dict['B_tensor'][ch].to(self.device) for ch in range(23)]
        else:
            self.real_B_tensor = input_dict['B_tensor'].to(self.device)
        self.data_index = input_dict['index']

    def forward(self):
        self.fake_A_tensor = self.netG(self.real_B_tensor)
        if self.param.zo_norm:
            self.fake_A_tensor = self.sigmoid(self.fake_A_tensor)
            self.fake_A_tensor = (self.param.target_max - self.param.target_min) * self.fake_A_tensor + self.param.target_min

    def test(self):
        with torch.no_grad():
            self.forward()

    def update(self):
        pass

    def init_fake_dict(self):
        fake_dict = OrderedDict()
        fake_dict['index'] = np.zeros(shape=(0,), dtype=np.int32)
        fake_dict['fake'] = np.zeros(shape=(0, self.param.output_chan_num), dtype=np.float32) # ← use output_chan_num
        print("[DEBUG][init_fake_dict] Initialized with keys:", fake_dict.keys())
        return fake_dict

    def update_fake_dict(self, fake_dict):
        """
        update the fake array that stores the predicted omics data
        fake_dict (OrderedDict)  -- the fake array that stores the predicted omics data and the index array
        """
        with torch.no_grad():
            if self.param.add_channel:
                current_fake_array = np.squeeze(self.fake_A_tensor.cpu().numpy(), axis=1)
            else:
                current_fake_array = self.fake_A_tensor.cpu().numpy()

            current_index_array = self.data_index.cpu().numpy()

            fake_dict['fake'] = np.concatenate((fake_dict['fake'], current_fake_array), axis=0)
            fake_dict['index'] = np.concatenate((fake_dict['index'], current_index_array), axis=0)

            return fake_dict
        