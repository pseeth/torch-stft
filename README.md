# STFT/iSTFT in PyTorch

An STFT/iSTFT written up in PyTorch using 1D Convolutions. Requirements are a recent version PyTorch, numpy, and librosa (for loading audio in test_stft.py). Thanks to Shrikant Venkataramani for sharing code this was based off of and Rafael Valle for catching bugs and adding the proper windowing logic. Uses Python 3.

Install easily with pip:
```
pip install torch_stft
```

Test it by just cloning this repo and run: 
    
```
pip install -r requirements.txt
python -m pytest .
```

Unfortunately, since it's implemented with 1D Convolutions, some filter_length/hop_length 
combinations can result in out of memory errors on your GPU when run on sufficiently large input.
