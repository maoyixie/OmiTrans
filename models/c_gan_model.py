### c_gan_model.py
import torch
from .basic_model import BasicModel
from . import networks
import torch.nn as nn

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
        return {'fake': []}

    def update_fake_dict(self, fake_dict):
        fake_np = self.fake_A_tensor.detach().cpu().numpy()
        if self.param.add_channel:
            fake_np = fake_np[0]
        fake_dict['fake'].append(fake_np)
        return fake_dict
