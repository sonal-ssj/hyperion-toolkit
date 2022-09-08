import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .utility import models


# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=16000, win=2, layer=8, stack=3, 
                 kernel=3, num_spk=2, causal=False, TCN_dilated=True, TCN_dilationFactor=2,
                 masks_type='mul', audio_scale=2**15-1, masking_nonlinearity='sigmoid', support_noise=False, dim_noise=32, std_noise=0.01):
        super(TasNet, self).__init__()
        
        # hyper parameters
        self.std_noise = std_noise
        self.dim_noise = dim_noise
        self.support_noise = support_noise
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim
        
        self.win = int(sr*win/1000)
        self.stride = self.win // 2
        
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.TCN_dilated = TCN_dilated
        self.TCN_dilationFactor = TCN_dilationFactor
        self.causal = causal
        self.masks_type = masks_type
        self.audio_scale = audio_scale
        self.masking_nonlinearity = masking_nonlinearity
        
        # input encoder
        self.encoder = nn.Conv1d(1, self.enc_dim - 1 if self.support_noise else self.enc_dim, self.win, bias=False, stride=self.stride)
        
        # TCN separator
        self.TCN = models.TCN(self.enc_dim, self.enc_dim*self.num_spk, self.feature_dim, self.feature_dim*4,
                              self.layer, self.stack, self.kernel, causal=self.causal, dilated=self.TCN_dilated, dilationFactor=self.TCN_dilationFactor)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input, z=None, return_enc_output=False):
        
        # padding
        output, rest = self.pad_signal(input)
        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.encoder(output)  # B, N, L
        if return_enc_output:
            return enc_output
        if z is None:
            if self.support_noise:
                z = torch.randn(enc_output.shape[0], self.dim_noise).to(enc_output)     # create noise in this case [validation / inference]
        if z is not None:
            assert self.support_noise
            assert len(z.shape) == 2    # B,E
            n_repeats = math.floor(enc_output.shape[-1] / z.shape[-1])
#            assert n_repeats > 0, f'{enc_output.shape=} {z.shape=}'
            if n_repeats > 0:
                z0 = z.repeat(1,n_repeats)
                rem = enc_output.shape[-1] % z.shape[-1]
                if rem == 0:
                    z = z0
                else:
                    z = torch.cat((z0,z[:,:rem]), dim=1)
            else:
                z = z[:,:enc_output.shape[-1]]
            z = z.unsqueeze(1)
            enc_output = torch.cat((enc_output, z), dim=1)

        # generate masks
        if self.masking_nonlinearity == 'sigmoid':
            mask_fn = torch.sigmoid
        elif self.masking_nonlinearity == 'tanh':
            mask_fn = torch.tanh
        else:
            raise NotImplementedError(f'{self.masking_nonlinearity}')
        masks = mask_fn(self.TCN(enc_output)).view(batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L
        if self.masks_type == 'mul':
            masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        elif self.masks_type == 'add':
            masked_output = enc_output.unsqueeze(1) + masks*(2**15-1)/self.audio_scale
        elif self.masks_type == 'muladd':
            masked_output = enc_output.unsqueeze(1)*masks + masks*(2**15-1)/self.audio_scale
        else:
            raise NotImplementedError(f'{self.masks_type}')
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_dim, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
        return output

def test_conv_tasnet():
    x = torch.rand(2, 32000)
    nnet = TasNet()
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
