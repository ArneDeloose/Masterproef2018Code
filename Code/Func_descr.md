# Decription of all the functions

---

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

set_freqthresh(num_class), return(min_freq, max_freq)

Sets the frequency of different bats for classification. Every bat has a number called num_class. Needed for the function compare_img.

---

spect(file_name), return(sample_rate, samples, t, total_time, steps, microsteps)

Reads a file and returns the necessary information to make a spectrogram.

---

spect_plot(samples, sample_rate),return(spectro)

Makes a spectrogram and converts it to grayscale values.

---

substraction(spect), return(spectro)

Subtracts the mean from all values in a spectrogram to reduce noise. Called within spect_plot.

---

spect_loop(file_name), return(rectangles2, regions2, spectros)

Extracts the ROIs from a file. This functions calls spect_plot, overload and ROI and ROI2. Spectrograms are created for every 200 ms and split in two. Regions are then extracted using ROI and ROI2.

Rectangles is a dictionary containing coordinates of all regions (stored as colums in the order: x, y of the lower left corner, width, height). Rectangles is labeled as 0,1,2,... (0: between 0 ms and 100 ms).

Regions is a dictionary following the same keys with the extracted regions as subdictionaries (labeled 0,1,2,...)

Spectros contains the full spectrograms (also stored as dictionary with numbered labels).

---

ROI(spect_norm, kern, X), return(ctrs, len_flag)

Extracts ROIs from an image. Spect_norm is the spectrogram, kern and X are called using set_parameters. Returns contours and a len_flag. If len_flag is falsem the image is empty and ROI2 is skipped.

---

ROI2(ctrs, spect_norm), return(Mask, regions)

Converts the contours to coordinates stored in Mask (mask is then assigned to the correct regions key), regions are extracted from the spectrogram (spect_norm). Only used if contours is not empty.

---

overload(rectangles, regions), return(rectangles2, regions2)

Deletes entries from rectangles and regions if there are more than max_roi (to reduce noise). Called within spect_loop.

---

show_region(rectangles, spectros, i), return()

Shows a plot of all the regions as red rectangles on the spectrogram. i is the time key to extract the correct spectrogram from the dictionary.

---

show_mregions(rectangles, spectros), return()

Shows the regions of the entire file one by one. This calls show_region for every i in order. Next spectrogram is shown when pressing enter.

---

compare_img(img1, img2, rectangle, min_freq, max_freq), return(score)

Calculates the SSIM between two images and returns it as a score. If the frequency isn't in the right range, score is set to the minimum (-1). For this, the function set_freqthresh is needed with the correct class number.

---

compare_img_plot(img1,img2), return()

Plots two images so they can be compared by hand. Used to visualise the SSIM.
