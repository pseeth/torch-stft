# STFT/iSTFT in PyTorch

An STFT/iSTFT written up in PyTorch using 1D Convolutions. Requirements are a recent version PyTorch, numpy, and librosa (for loading audio in test_stft.py). In older versions of PyTorch, conv1d_transpose may crash and this might not work. Also, thanks to Shrikant Venkataramani for sharing code this was based off of.


Test it by just cloning this repo and run: 
    
    python test_stft.py 

It should output the following:

    MSE: 5.6106e-18 @ filter_length = 1, hop_length = 1
    MSE: 6.82898e-19 @ filter_length = 2, hop_length = 1
    MSE: 5.11449e-18 @ filter_length = 2, hop_length = 2
    MSE: 1.16193e-19 @ filter_length = 4, hop_length = 1
    MSE: 5.87569e-19 @ filter_length = 4, hop_length = 2
    MSE: 4.34895e-18 @ filter_length = 4, hop_length = 4
    MSE: 1.92996e-19 @ filter_length = 8, hop_length = 1
    MSE: 1.21846e-18 @ filter_length = 8, hop_length = 2
    MSE: 4.17337e-18 @ filter_length = 8, hop_length = 4
    MSE: 1.66176e-17 @ filter_length = 8, hop_length = 8
    MSE: 7.76624e-19 @ filter_length = 16, hop_length = 1
    MSE: 3.46508e-18 @ filter_length = 16, hop_length = 2
    MSE: 1.09163e-17 @ filter_length = 16, hop_length = 4
    MSE: 2.68046e-17 @ filter_length = 16, hop_length = 8
    MSE: 5.46612e-17 @ filter_length = 16, hop_length = 16
    MSE: 2.05615e-18 @ filter_length = 32, hop_length = 1
    MSE: 6.10532e-18 @ filter_length = 32, hop_length = 2
    MSE: 1.6901e-17 @ filter_length = 32, hop_length = 4
    MSE: 3.88015e-17 @ filter_length = 32, hop_length = 8
    MSE: 7.38627e-17 @ filter_length = 32, hop_length = 16
    MSE: 1.35141e-16 @ filter_length = 32, hop_length = 32
    MSE: 3.04917e-18 @ filter_length = 64, hop_length = 1
    MSE: 7.78764e-18 @ filter_length = 64, hop_length = 2
    MSE: 1.95982e-17 @ filter_length = 64, hop_length = 4
    MSE: 4.64905e-17 @ filter_length = 64, hop_length = 8
    MSE: 8.92093e-17 @ filter_length = 64, hop_length = 16
    MSE: 1.57783e-16 @ filter_length = 64, hop_length = 32
    MSE: 2.97127e-16 @ filter_length = 64, hop_length = 64
    MSE: 3.55324e-18 @ filter_length = 128, hop_length = 1
    MSE: 8.01636e-18 @ filter_length = 128, hop_length = 2
    MSE: 1.94888e-17 @ filter_length = 128, hop_length = 4
    MSE: 4.92468e-17 @ filter_length = 128, hop_length = 8
    MSE: 1.04113e-16 @ filter_length = 128, hop_length = 16
    MSE: 1.8359e-16 @ filter_length = 128, hop_length = 32
    MSE: 3.27309e-16 @ filter_length = 128, hop_length = 64
    MSE: 6.35962e-16 @ filter_length = 128, hop_length = 128
    MSE: 3.42083e-18 @ filter_length = 256, hop_length = 1
    MSE: 7.63106e-18 @ filter_length = 256, hop_length = 2
    MSE: 1.8438e-17 @ filter_length = 256, hop_length = 4
    MSE: 4.69986e-17 @ filter_length = 256, hop_length = 8
    MSE: 1.13181e-16 @ filter_length = 256, hop_length = 16
    MSE: 2.13975e-16 @ filter_length = 256, hop_length = 32
    MSE: 3.74273e-16 @ filter_length = 256, hop_length = 64
    MSE: 6.66832e-16 @ filter_length = 256, hop_length = 128
    MSE: 1.28269e-15 @ filter_length = 256, hop_length = 256
    MSE: 3.20702e-18 @ filter_length = 512, hop_length = 1
    MSE: 7.23558e-18 @ filter_length = 512, hop_length = 2
    MSE: 1.72297e-17 @ filter_length = 512, hop_length = 4
    MSE: 4.48844e-17 @ filter_length = 512, hop_length = 8
    MSE: 1.17124e-16 @ filter_length = 512, hop_length = 16
    MSE: 2.47271e-16 @ filter_length = 512, hop_length = 32
    MSE: 4.4375e-16 @ filter_length = 512, hop_length = 64
    MSE: 7.86491e-16 @ filter_length = 512, hop_length = 128
    MSE: 1.40834e-15 @ filter_length = 512, hop_length = 256
    MSE: 2.76763e-15 @ filter_length = 512, hop_length = 512
    MSE: 1.51821e-17 @ filter_length = 1024, hop_length = 1
    MSE: 6.71528e-18 @ filter_length = 1024, hop_length = 2
    MSE: 1.57908e-17 @ filter_length = 1024, hop_length = 4
    MSE: 4.00848e-17 @ filter_length = 1024, hop_length = 8
    MSE: 1.1863e-16 @ filter_length = 1024, hop_length = 16
    MSE: 2.86171e-16 @ filter_length = 1024, hop_length = 32
    MSE: 5.3835e-16 @ filter_length = 1024, hop_length = 64
    MSE: 9.51598e-16 @ filter_length = 1024, hop_length = 128
    MSE: 1.62187e-15 @ filter_length = 1024, hop_length = 256
    MSE: 2.88709e-15 @ filter_length = 1024, hop_length = 512
    MSE: 5.67381e-15 @ filter_length = 1024, hop_length = 1024
    MSE: 1.13245e-15 @ filter_length = 2048, hop_length = 1
    MSE: 1.38272e-17 @ filter_length = 2048, hop_length = 2
    MSE: 1.42272e-17 @ filter_length = 2048, hop_length = 4
    MSE: 3.62482e-17 @ filter_length = 2048, hop_length = 8
    MSE: 1.07733e-16 @ filter_length = 2048, hop_length = 16
    MSE: 3.17108e-16 @ filter_length = 2048, hop_length = 32
    MSE: 6.63297e-16 @ filter_length = 2048, hop_length = 64
    MSE: 1.12009e-15 @ filter_length = 2048, hop_length = 128
    MSE: 1.89827e-15 @ filter_length = 2048, hop_length = 256
    MSE: 3.25905e-15 @ filter_length = 2048, hop_length = 512
    MSE: 5.68223e-15 @ filter_length = 2048, hop_length = 1024
    MSE: 1.10644e-14 @ filter_length = 2048, hop_length = 2048
    
Unfortunately, since it's implemented with 1D Convolutions, some filter_length/hop_length combinations can result in out of memory errors on your GPU when run on sufficiently large input.
