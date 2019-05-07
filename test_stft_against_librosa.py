import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import librosa
from matplotlib import pyplot as plt

import stft

def get_last_dim(x, d=1):
    while len(x.shape) > d:
        x = x[0, ...]
    return x

def enhance_for_plot(x):
    if type(x) == torch.Tensor:
        x = x.numpy()
    x = get_last_dim(x, d=2)
    return np.flip((np.log(x+1)*10)**.33, 0)

n_fft = 1024
hop_length = 512
filter_length = n_fft
window = torch.hamming_window(n_fft)

audio, sr = librosa.load(librosa.util.example_audio_file())
audio_tensor = torch.FloatTensor(audio)[None, :] # wrap in tensor and add empty batch dimension

# Determine Spectrograms, Phases and Angles
# librosa
librosa_stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
librosa_spect, librosa_phase = librosa.magphase(librosa_stft)
librosa_angle = np.angle(librosa_phase)

# PyTorch's own stft (is based on pseeth's original?)
pytorch_stft = torch.stft(audio_tensor, n_fft=n_fft, hop_length=hop_length, window=window)
pytorch_spect = torch.sqrt(torch.sum(pytorch_stft**2, dim=-1))
pytorch_angle = torch.atan2(pytorch_stft[..., 1], pytorch_stft[..., 0])

# Pseeth's implementation (augmented with windowing fn)
stft_obj = stft.STFT(filter_length=filter_length, hop_length=hop_length, window=window)
pseeth_spect, pseeth_angle = stft_obj.transform(audio_tensor)

# Show that audio -stft-> spect -istft-> audio makes proper reconstructions with pseeths's istft (librosa-pseeth has a minor quality degradation though)
# librosa-librosa
recon = librosa.istft(librosa_spect * librosa_phase, hop_length=hop_length)
librosa.output.write_wav('librosa-stft_librosa-istft.wav', recon, sr)

# pseeth-pseeth
recon = stft_obj.inverse(pseeth_spect, pseeth_angle)
recon = get_last_dim(recon, d=1).numpy()
librosa.output.write_wav('pseeth-stft_pseeth-istft.wav', recon, sr)

# pytorch-pseeth
recon = stft_obj.inverse(pytorch_spect, pytorch_angle)
recon = get_last_dim(recon, d=1).numpy()
librosa.output.write_wav('pytorch-stft_pseeth-istft.wav', recon, sr)

# librosa-pseeth
librosa_spect_tensor, librosa_angle_tensor = torch.FloatTensor(librosa_spect)[None, ...], torch.FloatTensor(librosa_angle)[None, ...] # wrap in tensor and add empty batch dimension
recon = stft_obj.inverse(librosa_spect_tensor, librosa_angle_tensor)
recon = get_last_dim(recon, d=1).numpy()
librosa.output.write_wav('librosa-stft_pseeth-istft.wav', recon, sr)

# Plots and Figures
plt.subplot(311)
plt.title("Spectrograms by Librosa, PyTorch, PSeeth")
plt.imshow(enhance_for_plot(librosa_spect))
plt.subplot(312)
plt.imshow(enhance_for_plot(pytorch_spect))
plt.subplot(313)
plt.imshow(enhance_for_plot(pseeth_spect))
plt.savefig('Spects_librosa_pytorch_pseeth.png')
plt.show()
