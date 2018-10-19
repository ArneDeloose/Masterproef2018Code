# Decription of all the functions

set_parameters(), return(X, kern, thresh, max_roi, min_spec_freq, max_spec_freq)  

This function sets a number of parameters needed in other functions. It is called automatically in other functions when needed. Keeping all parameters in one place allows for easier tuning.  

The parameters are:

*X: threshold for binary image (number between 0 and 256)
*kern: kernel size for ROI
*thresh: threshold for SSIM
*max_roi: maximum number of regions within one image (100 ms)
*min_spec_freq: lowest frequency in the spectrogram (1 point =0.375 kHz)
*max_spec_freq: highest frequency in the spectrogram

---

set_freqthresh(), return(min_freq, max_freq)

Sets the frequency of different bats for classification

---

spect(file_name): return(sample_rate, samples, t, total_time, steps, microsteps)

Reads a file and returns the necessary information to make a spectrogram.


