# Audio tools for PyTorch

Right now, just an STFT/iSTFT written up in PyTorch using 1D Convolutions. Requirements are a recent version PyTorch, numpy, and librosa (for loading audio in test_stft.py). In older versions of PyTorch, conv1d_transpose may crash and this might not work.


Test it by just cloning this repo and run: 
    
    python test_stft.py 

It should output the following:

    MSE: 6.52442e-19 @ filter_length = 2, hop_length = 1
    MSE: 1.12645e-19 @ filter_length = 4, hop_length = 1
    MSE: 5.83806e-19 @ filter_length = 4, hop_length = 2
    MSE: 1.9048e-19 @ filter_length = 8, hop_length = 1
    MSE: 1.2711e-18 @ filter_length = 8, hop_length = 2
    MSE: 4.1606e-18 @ filter_length = 8, hop_length = 4
    MSE: 7.96194e-19 @ filter_length = 16, hop_length = 1
    MSE: 3.51195e-18 @ filter_length = 16, hop_length = 2
    MSE: 1.10745e-17 @ filter_length = 16, hop_length = 4
    MSE: 2.64801e-17 @ filter_length = 16, hop_length = 8
    MSE: 2.09022e-18 @ filter_length = 32, hop_length = 1
    MSE: 6.14735e-18 @ filter_length = 32, hop_length = 2
    MSE: 1.72562e-17 @ filter_length = 32, hop_length = 4
    MSE: 3.9354e-17 @ filter_length = 32, hop_length = 8
    MSE: 7.32921e-17 @ filter_length = 32, hop_length = 16
    MSE: 3.08971e-18 @ filter_length = 64, hop_length = 1
    MSE: 8.02838e-18 @ filter_length = 64, hop_length = 2
    MSE: 1.93203e-17 @ filter_length = 64, hop_length = 4
    MSE: 4.61839e-17 @ filter_length = 64, hop_length = 8
    MSE: 8.89388e-17 @ filter_length = 64, hop_length = 16
    MSE: 1.58822e-16 @ filter_length = 64, hop_length = 32
    MSE: 3.46813e-18 @ filter_length = 128, hop_length = 1
    MSE: 8.09248e-18 @ filter_length = 128, hop_length = 2
    MSE: 1.96637e-17 @ filter_length = 128, hop_length = 4
    MSE: 4.98187e-17 @ filter_length = 128, hop_length = 8
    MSE: 1.03765e-16 @ filter_length = 128, hop_length = 16
    MSE: 1.83023e-16 @ filter_length = 128, hop_length = 32
    MSE: 3.31214e-16 @ filter_length = 128, hop_length = 64
    MSE: 3.50962e-18 @ filter_length = 256, hop_length = 1
    MSE: 7.82393e-18 @ filter_length = 256, hop_length = 2
    MSE: 1.8228e-17 @ filter_length = 256, hop_length = 4
    MSE: 4.75668e-17 @ filter_length = 256, hop_length = 8
    MSE: 1.13521e-16 @ filter_length = 256, hop_length = 16
    MSE: 2.12114e-16 @ filter_length = 256, hop_length = 32
    MSE: 3.72732e-16 @ filter_length = 256, hop_length = 64
    MSE: 6.78565e-16 @ filter_length = 256, hop_length = 128
    MSE: 3.24858e-18 @ filter_length = 512, hop_length = 1
    MSE: 7.36809e-18 @ filter_length = 512, hop_length = 2
    MSE: 1.71541e-17 @ filter_length = 512, hop_length = 4
    MSE: 4.42971e-17 @ filter_length = 512, hop_length = 8
    MSE: 1.18888e-16 @ filter_length = 512, hop_length = 16
    MSE: 2.48935e-16 @ filter_length = 512, hop_length = 32
    MSE: 4.46269e-16 @ filter_length = 512, hop_length = 64
    MSE: 8.01727e-16 @ filter_length = 512, hop_length = 128
    MSE: 1.41943e-15 @ filter_length = 512, hop_length = 256
    MSE: 9.7475e-18 @ filter_length = 1024, hop_length = 1
    MSE: 6.97216e-18 @ filter_length = 1024, hop_length = 2
    MSE: 1.5919e-17 @ filter_length = 1024, hop_length = 4
    MSE: 3.98846e-17 @ filter_length = 1024, hop_length = 8
    MSE: 1.19764e-16 @ filter_length = 1024, hop_length = 16
    MSE: 2.86896e-16 @ filter_length = 1024, hop_length = 32
    MSE: 5.37956e-16 @ filter_length = 1024, hop_length = 64
    MSE: 9.88971e-16 @ filter_length = 1024, hop_length = 128
    MSE: 1.65615e-15 @ filter_length = 1024, hop_length = 256
    MSE: 2.89969e-15 @ filter_length = 1024, hop_length = 512
    MSE: 1.48655e-15 @ filter_length = 2048, hop_length = 1
    MSE: 1.17908e-17 @ filter_length = 2048, hop_length = 2
    MSE: 1.45543e-17 @ filter_length = 2048, hop_length = 4
    MSE: 3.70206e-17 @ filter_length = 2048, hop_length = 8
    MSE: 1.08357e-16 @ filter_length = 2048, hop_length = 16
    MSE: 3.11991e-16 @ filter_length = 2048, hop_length = 32
    MSE: 6.7189e-16 @ filter_length = 2048, hop_length = 64
    MSE: 1.15825e-15 @ filter_length = 2048, hop_length = 128
    MSE: 1.95168e-15 @ filter_length = 2048, hop_length = 256
    MSE: 3.26432e-15 @ filter_length = 2048, hop_length = 512
    MSE: 5.74241e-15 @ filter_length = 2048, hop_length = 1024
    
Unfortunately, since it's implemented with 1D Convolutions, some filter_length/hop_length combinations can result in out of memory errors on your GPU when run on sufficiently large input.
