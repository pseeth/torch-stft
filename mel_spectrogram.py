import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import librosa
from stft import STFT
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

class MelSpectrogram(torch.nn.Module):
    """
        Example:
            audio, sr = librosa.load("mixture.mp3", sr=None)
            audio = Variable(torch.FloatTensor(audio), requires_grad=False).unsqueeze(0)
            mel_transform = MelSpectrogram(sample_rate=sr, filter_length=1024, num_mels=150)
            mel_spectrogram = mel_transform(audio).squeeze(0).data.numpy()
        """
    def __init__(self, sample_rate=44100, filter_length=1024, hop_length=512, num_mels=150):
        super(MelSpectrogram, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate  
        
        self.stft = STFT(filter_length=self.filter_length, hop_length=self.hop_length)
        mel_filters = librosa.filters.mel(self.sample_rate, self.filter_length, self.num_mels)
        self.mel_filter_bank = Variable(torch.FloatTensor(mel_filters), requires_grad=False)       

    def forward(self, input_data):
        magnitude, phase = self.stft.transform(input_data)
        mel_spectrogram = F.linear(magnitude.transpose(-1, -2), self.mel_filter_bank)
        mel_spectrogram = 10.0 * (torch.log(mel_spectrogram**2 + 1e-8) / np.log(10.0))
        return mel_spectrogram