import librosa
import traceback
from torch_stft import STFT
import torch
import numpy as np
    
def _test_stft_on_signal(input_audio, atol, device):
    audio = torch.FloatTensor(input_audio)
    if len(audio.shape) < 2:
        audio = audio.unsqueeze(0)

    audio = audio.to(device)

    for i in range(10):
        filter_length = 2**i
        for j in range(i):
            hop_length = 2**j
            stft = STFT(
                filter_length=filter_length, 
                hop_length=hop_length, 
                win_length=filter_length
            ).to(device)
            output = stft(audio)
            output = output.cpu().data.numpy()[..., :]
            _audio = audio.cpu().data.numpy()[..., :]
            assert (np.mean((output - _audio) ** 2) < atol)

def test_stft():
    # White noise
    test_audio = []
    # seed = np.random.RandomState(0)
    # x1 = seed.randn(2 ** 15)
    # test_audio.append((x1, 1e-6))

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
        # x1 = seed.randn(2 ** 15)
        # batched.append(x1)
        # Sin wave
        x2 = np.sin(np.linspace(-np.pi, np.pi, 2 ** 15))
        batched.append(x2)

    device = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']
    batched = np.vstack(batched)
    for d in device:
        _test_stft_on_signal(batched, 1e-10, d)