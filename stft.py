import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=512):
        super(STFT, self).__init__()
        
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))                    
        
        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), 
                                   np.imag(fourier_basis[:cutoff, :])])
        
        self.forward_basis = nn.Parameter(torch.FloatTensor(fourier_basis[:, None, :]), 
                                                requires_grad = False)
        self.inverse_basis = nn.Parameter(torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :]), 
                                                requires_grad = False)
                
    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)
        
        self.num_samples = num_samples
                
        input_data = input_data.view(num_batches, 1, num_samples)
        forward_transform = F.conv1d(input_data, 
                                     self.forward_basis, 
                                     stride = self.hop_length, 
                                     padding = self.filter_length)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase        
    
    def inverse(self, magnitude, phase): 
        recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase), 
                                                   magnitude*torch.sin(phase)], dim=1)
            
        inverse_transform = F.conv_transpose1d(recombine_magnitude_phase,
                                               self.inverse_basis,
                                               stride=self.hop_length,
                                               padding=0)
        inverse_transform = inverse_transform[:, :, self.filter_length:]
        inverse_transform = inverse_transform[:, :, :self.num_samples]
        return inverse_transform
    
    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction