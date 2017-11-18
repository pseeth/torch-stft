import librosa
import traceback
from stft import STFT
import torch
    
def test_stft():
    audio, sr = librosa.load('mixture.mp3', sr=None)
    audio = torch.autograd.Variable(torch.FloatTensor(audio), requires_grad=False).unsqueeze(0)
    if torch.cuda.is_available():
        audio = audio.cuda()
    
    def mse(ground_truth, estimated):
        return torch.mean((ground_truth - estimated)**2)

    def to_np(tensor):
        return tensor.cpu().data.numpy()

    for i in range(12):
        filter_length = 2**i
        for j in range(i):
            try:
                hop_length = 2**j
                stft = STFT(filter_length=filter_length, hop_length=hop_length)
                if torch.cuda.is_available():
                    stft = stft.cuda()
                output = stft(audio)
                loss = mse(output, audio)
                print 'MSE: %s @ filter_length = %d, hop_length = %d' % (str(to_np(loss)[0]), filter_length, hop_length)
            except:
                print 'Failed @ filter_length = %d, hop_length = %d' % (filter_length, hop_length)
                print traceback.print_exc()

if __name__ == "__main__":
    test_stft()