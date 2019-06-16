import scipy.io.wavfile

file_name='C:/Users/arne/Documents/School/Thesis/Audio_data/ppip_test_stereo.wav';
sample_rate, samples=scipy.io.wavfile.read(file_name, mmap=False)

import librosa

samples2, sample_rate2= librosa.load(file_name, sr=48000, mono=False)
samples2= samples2 * 32768 #converting factor