# Decription of all the functions

---

set_parameters(), return(X, kern, thresh, max_roi, min_spec_freq, max_spec_freq)  

This function sets a number of parameters needed in other functions. It is called automatically in other functions when needed. Keeping all parameters in one place allows for easier tuning.  

The parameters are:

* X: threshold for binary image (number between 0 and 256)

* kern: kernel size for ROI

* thresh: threshold for SSIM

* max_roi: maximum number of regions within one image (100 ms)

* min_spec_freq: lowest frequency in the spectrogram (1 point =0.375 kHz)

* max_spec_freq: highest frequency in the spectrogram

---

set_freqthresh(num_class), return(min_freq, max_freq)

Sets the frequency of different bats for classification. Every bat has a number called num_class. Needed for the function compare_img.

---

spect(file_name, \**optional), return(sample_rate, samples, t, total_time, steps, microsteps)

Reads a file and returns the necessary information to make a spectrogram. For time dilation spectrograms an optinional argument 'channel' needs to be given which needs to be 'r' or 'l' (right or left channel). Steps and microsteps give information about the number of spectrograms that will be created.

---

spect_plot(samples, sample_rate),return(spectro)

Makes a spectrogram and converts it to grayscale values. Information from spect is needed.

---

substraction(spect), return(spectro)

Subtracts the mean from all values in a spectrogram to reduce noise. Called within spect_plot.

---

spect_loop(file_name, \**optional), return(rectangles2, regions2, spectros)

Extracts the ROIs from a file. This functions calls set_parameters, spect, spect_plot, overload and ROI and ROI2. Spectrograms are created every 200 ms and split in two. Regions are then extracted using ROI and ROI2. If the file used time dilation, an optional argument 'channel' needs to be given which is 'r' or 'l' (left or right channel).

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

---

create_smatrix(rectangles, spectros, num_classes), return(s_mat)

Creates an empty score matrix to store the SSIMs for every regio.

---

calc_smatrix(s_mat, regions, rectangles, templates, num), return(s_mat2)

Calculates the score matrix. Must be called again for every class and needs templates and an empty score matrix.

---

create_cmatrix(rectangles, spectros), return(c_mat)

Creates an empty classification matrix to store the label for every regio.

---

calc_cmatrix(c_mat, s_mat), return(c_mat2)

Fills the c matrix. Uses the threshold from set_parameters and the score matrix.

Classes:

* 0: empty

* 1: non-classified

* n-2: class 0

---

calc_result(c_mat, num_classes), return(res)

Calculates a result matrix counting the instances of every class in c_mat.

---

loop_res(rectangles, spectros, regions, templates), return(res, c_mat, s_mat)

Pools the functions create_smatrix, calc_smatrix, create_cmatrix, calc_cmatrix and calc_result.

---

show_class(class_num, c_mat, rectangles, regions, spectros), return()

Shows all regions from a class one by one. Press enter for a new plot.

---

loop_full(file_name), return(res)

Uses spect_loop, create_template_set and loop_res to analyse a full sound returning only the result.

---

 calc_sim_matrix(rectangles, regions), return(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6)

 Compares all possible pairings between two regions from a given dictionary of regions and rectangles (rectangles contains freq info).
 Comparison returns six matrices with information: ssim, squared difference in frequency range, min freq, max freq, mean freq and duration.

---

calc_dist_matrix(sim_mat1, sim_mat2, sim_mat3, sim_mat4, sim_mat5, sim_mat6, weight), return(dist_mat)

Calculates a total distance based on the similarity measures calculated in 'calc_sim_matrix'. Requires a weighing of all measures. Weights are set in 'set_weights' (which is called automatically).

---

set_weights(weight), return(w1,w2,w3,w4,w5,w6)

Sets weights for 'calc_dist_matrix'. The variable 'weight' allows you to drop one weight from the set to study the importance of it. If weight=0 nothing is dropped. Standard weights are set at roughly 1/mean(sim)

---

calc_pos(dist_mat), return(pos)

Calculates the position in two dimensions from the distance matrix. For this, an MDS is used.

---

calc_pos_TSNE(dist_mat), return(pos)

Similar to 'calc_pos' but uses the TSNE instead of the MDS.

---

plot_MDS(pos), return()

Plots an MDS or TSNE from the given positions. This is calibrated to use labels corresponding to the set in 'create_template_set'.

---

run_MDS(weight), return()

Pools the functions: 'set_templates2', 'calc_sim_matrix', 'calc_dist_matrix', 'calc_pos' and 'plot_MDS'.

---

run_TSNE, return()

Analogous to 'run_MDS' but with the TSNE.

---

set_templates2(), return(rectangles_final, regions_final)

Extracts pulses from five different bats (ppip, eser, mdau, pnat and nlei) to use as reference images.

---

calc_features(rectangles, regions, templates, num_reg), return(features, features_key, features_freq)

Transforms regions into a feature matrix. Every row is one region. The first five columns contain frequency information (freq range, min freq, max freq, av freq, duration). These are scaled to contain half of the information in total divided equally over the five metrics. The remaining columns are the ssim scores with every reference in 'templates' (set by 'set_templates2'). Num_reg is the total number of regions to allow pre-allocation (can be calculated by 'calc_num_regions').

Features_key links the row number to a position in the original dictionary and features_freq are the first five columns unscaled.

---

calc_num_regions(regions), return(num_reg)

Calculates number of regions in a dictionary.

---

calc_col_labels(features), return(label_colors)

Labels every sound in features based on the maximum ssim in the row. If this max ssim is above a threshold (set by 'set_parameters' automatically) the pulse is labeled to be this species. Labels are stored as a dictionary with the row number as key containing a python coded color.

---

calc_col_labels2(features, features_freq), return(label_colors)

More advanced version of 'calc_col_labels'. Instead of only the maximum ssim, it takes all ssims above a threshold and adds them up to calculate a percentage matching (so if 20 of the 40 ppip ssim scores are above the threshold, the score is 50 %). Scores are weighed according to the squared difference between the lowest frequency and a fixed literature value for a certain bat. These values are called automatically using the function 'set_batfreq'. Number of templates per bat are called using 'set_numbats'.

---

set_numbats(), return(num_total, num_ppip, num_eser, num_mdau, num_pnat, num_nlei)

Sets the number of reference images for each bat. Called in 'calc_col_labels2'.

---

set_batfreq(), return(freq_ppip, freq_eser, freq_mdau, freq_pnat, freq_nlei)

Sets the lowest frequencies for each bat. Called in 'calc_col_labels2'.

---

plot_dendrogram(features, label_colors), return()

Plots a dendrogram based upon a hierarchical clustering of the features. Row numbers are colored based on label_colors. Linkage is set as 'average'.

---

show_region2(rectangles, spectros, features_key, i), return()

Shows a plot of a region on a spectrogram based on the number in the feature data (i). The title of the plot gives the frequency range of the sound and the timestep.

---

hier_clustering(file_name), return()

Function that pools various functions together to create a dendrogram right away from the name of a file. (Functions called: spect_loop, calc_num_regions, set_templates2, calc_features, calc_col_labels2, plot_dendrogram).
