import librosa
import traceback
from torch_stft import STFT
import torch
import numpy as np
from torch.nn import MSELoss

def _prepare_audio(input_audio, device):
    audio = torch.FloatTensor(input_audio)
    if len(audio.shape) < 2:
        audio = audio.unsqueeze(0)
    audio = audio.to(device)
    return audio

def _prepare_network(device, filter_length=1024, hop_length=512, win_length=None, window='hann'):
    stft = STFT(
            filter_length=filter_length, 
            hop_length=hop_length, 
            win_length=win_length,
            window=window
        ).to(device)
    return stft
    
def _test_stft_on_signal(input_audio, atol, device):
    audio = _prepare_audio(input_audio, device)    
    for i in range(10):
        filter_length = 2**i
        for j in range(i):
            hop_length = 2**j
            stft = _prepare_network(device, filter_length, hop_length)
            output = stft(audio)
            output = output.cpu().data.numpy()[..., :]
            _audio = audio.cpu().data.numpy()[..., :]
            assert (np.mean((output - _audio) ** 2) < atol)

def test_stft():
    # White noise
    test_audio = []
    seed = np.random.RandomState(0)
    x1 = seed.randn(2 ** 15)
    test_audio.append((x1, 1e-10))

    # Sin wave
    x2 = np.sin(np.linspace(-np.pi, np.pi, 2 ** 15))
    test_audio.append((x2, 1e-10))

    # Music file
    x3 = librosa.load(librosa.util.example_audio_file(), duration=1.0)[0]
    test_audio.append((x3, 1e-10))
    device = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']

    for x, atol in test_audio:
        for d in device:
            _test_stft_on_signal(x, atol, d)

def test_batch_stft():
    # White noise
    seed = np.random.RandomState(0)
    batched = []
    batch_size = 10
    for i in range(int(batch_size / 2)):
        x1 = seed.randn(2 ** 15)
        batched.append(x1)
        # Sin wave
        x2 = np.sin(np.linspace(-np.pi, np.pi, 2 ** 15))
        batched.append(x2)

    device = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    batched = np.vstack(batched)
    for d in device:
        _test_stft_on_signal(batched, 1e-10, d)

def test_against_librosa_stft():
    audio = librosa.load(librosa.util.example_audio_file(), duration=10.0, offset=30)[0]
    for i in range(8, 12):
        filter_length = 2**i
        for j in range(4, i):
            hop_length = 2**j
            librosa_stft = librosa.stft(audio, n_fft=filter_length, hop_length=hop_length)
            _magnitude = np.abs(librosa_stft)
            _phase = np.arctan2(librosa_stft.imag, librosa_stft.real)
            device = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
            for d in device:
                _audio = _prepare_audio(audio, d)
                stft = _prepare_network(d, filter_length, hop_length)
                magnitude, phase = stft.transform(_audio)
                magnitude = magnitude[0].cpu().data.numpy()
                # commented out but can visualize the stft if needed
                # import matplotlib.pyplot as plt
                # plt.subplot(211)
                # plt.imshow(20*np.log10(1+magnitude), aspect='auto', origin='lower')
                # plt.subplot(212)
                # plt.imshow(20*np.log10(1+_magnitude), aspect='auto', origin='lower')
                # plt.show()
                phase = phase[0].cpu().data.numpy()

                assert (np.mean((magnitude - _magnitude) ** 2) < 1e-10)
                assert (np.mean((np.abs(phase) - np.abs(_phase)) ** 2) < 1e-2)

def test_stft_defaults():
    audio = librosa.load(librosa.util.example_audio_file(), duration=1.0)[0]
    device = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    for d in device:
        _audio = _prepare_audio(audio, d)
        stft = _prepare_network(d)
        output = stft.forward(_audio)
        output = output.cpu().data.numpy()[..., :]
        _audio = _audio.cpu().data.numpy()[..., :]
        assert (np.mean((output - _audio) ** 2) < 1e-10)

def test_windows():
    windows = ['bartlett', 'hann', 'hamming', 'blackman', 'blackmanharris']
    audio = librosa.load(librosa.util.example_audio_file(), duration=1.0)[0]
    device = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    for d in device:
        for w in windows:
            _audio = _prepare_audio(audio, d)
            stft = _prepare_network(d, window=w)
            output = stft.forward(_audio)
            output = output.cpu().data.numpy()[..., :]
            _audio = _audio.cpu().data.numpy()[..., :]
            assert (np.mean((output - _audio) ** 2) < 1e-10)

def dummy_network(input_shape):
    network = torch.nn.Sequential(
        torch.nn.Linear(input_shape[-1], input_shape[-1])
    )
    return network

def test_backward():
    audio = np.sin(np.linspace(-np.pi, np.pi, 2 ** 10))
    device = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    loss_function = MSELoss()
    n_iter = 1000
    for d in device:
        _audio = _prepare_audio(audio, d)
        input_network = dummy_network(_audio.shape)
        optimizer = torch.optim.SGD(input_network.parameters(), lr=1e-1)
        stft = _prepare_network(d, filter_length=256, hop_length=64)
        for i in range(n_iter):
            output = input_network(_audio)
            output = stft.forward(output)
            loss = loss_function(_audio, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        assert (loss.item() < 1e-7)