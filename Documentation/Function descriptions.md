Decription of all the functions. Use ctrl+F to find a function

**Module 1: AD1_Loading:**

---

loading_init(\**optional), return(freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats, list_bats, colors_bat, num_bats, num_total, regions_temp, rectangles_temp)

Loads several important variables. Called automatically in several functions.

---

set_parameters(), return(para)  

This function sets a number of parameters needed in other functions. It reads in these parameters from the textfile 'parameters.txt'. This function is called automatically in other functions when needed. Keeping all parameters in one place allows for easier tuning.  Parameters can be overwritten with an optional argument in other functions.

---

set_path(), return(path)

Sets the path as the current working directory (needed to read and write files). Called whenever needed.

---

read_templates(\**optional), return(regions, rectangles)

Reads in the templates from the correct folder. If the folder is somewhere else, this can be specified with the optional 'Templates' argument.

---

make_folders(path, list_bats), return()

Checks whether necessary folders exist for templates and creates new ones if need be.

---

import_map(map_name, \**optional), return(net, raw_data)

Imports a SOM map. Maps are stored as 'map_name'.npy and 'map_name'\_data.npy. Raw_data can be used to visualise the map on the data that was used. Optional argument 'path' can be used to specify a different directory.

---

import_map(dml, \**optional), return(D)

Imports a dml matrix. Optional argument 'path' can be used to specify a different directory.

---

set_numbats(list_bats, \**optional), return(num_bats, num_total)

Sets the number of reference images for each bat. For this the code assumes there is a folder 'Templates_arrays' in the working directory. If this is not the case, an alternate path can be specified as the optional argument 'Templates'.

---

set_batfreq(rectangles_temp, regions_temp, list_bats, num_bats), return(freq_bats, freq_range_bats, freq_peakT_bats, freq_peakF_bats)

Sets a number of frequencies needed for the calculation of the scores (see cacl_col_labels). For this, the median value over the references is taken.

---

set_batscolor(\**optional), return(list_bats, colors_bat)

Assigns a color to each bat. The optional argument 'Templates' can be used to specify an alternate path to the templates.

---

print_features(\**optional), return()

Prints out which feature index correlates to which bat (or frequency).

---

**Module 2: AD2_Spectro:**

---

spect(file_name, \**optional), return(sample_rate, samples, t, total_time, steps, microsteps)

Reads a file and returns the necessary information to make a spectrogram. For time dilation spectrograms an optinional argument 'channel' needs to be given which needs to be 'r' or 'l' (right or left channel). Steps and microsteps give information about the number of spectrograms that will be created.

---

spect_plot(samples, sample_rate, \**optional),return(spectro)

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

check_overlap(rectangles, regions, spectros, i, spect_window, spect_overlap_window), return(rectangles1, spectros1)

Deletes a region when there is an overlap with the previous spectrogram.

---

rescale_region(reg), return(regions)

Rescales a region to a range 0-255

---

ROI(spect_norm, kern, X), return(ctrs, len_flag)

Extracts ROIs from an image. Spect_norm is the spectrogram, kern and X are called using set_parameters (or specified in the previous function). Returns contours and a len_flag. If len_flag is falsem the image is empty and ROI2 is skipped.

---

ROI2(ctrs, spect_norm), return(Mask, regions)

Converts the contours to coordinates stored in Mask (mask is then assigned to the correct regions key), regions are extracted from the spectrogram (spect_norm). Only used if contours is not empty.

---

overload(rectangles, regions, \**optional), return(rectangles2, regions2)

Deletes entries from rectangles and regions if there are more than max_roi (to reduce noise). Called within spect_loop.

---

show_region(rectangles, spectros, i, \**optional), return()

Shows a plot of all the regions as red rectangles on the spectrogram. i is the time key to extract the correct spectrogram from the dictionary. If a path is given in optional, the image is saved to this path.

---

show_mregions(rectangles, spectros), return()

Shows the regions of the entire file one by one. This calls show_region for every i in order. Next spectrogram is shown when pressing enter.

---

wave_plot(data, wavelet), return(fig)

Plots a wavelet. Input 'wavelet' is a string with the name of the wavelet

---

create_template(file_name, timestep, region_num, bat_name, \**optional), return()

Creates a template (array, image and rectangle). The name of the file must be given, along with the timestep and the region number. The code name (eser, ppip, pnat,...) must be given. If this bat doesn't exist yet, folders are created. The optional argument 'Templates' can be given to specify an alternate path to the templates folders. Hash numbers for the image and array are printed out.

---

**Module 3: AD3_Features:**

---

calc_features(rectangles, regions, templates, num_reg, list_bats, num_total), return(features, features_key, features_freq)

Transforms regions into a feature matrix. Every row is one region. The first seven columns contain frequency information (freq range, min freq, max freq, av freq, duration, peak T freq and peak F freq). These are scaled to contain half of the information in total divided equally over the seven metrics. The remaining columns are the ssim scores with every reference in 'templates'. Num_reg, list_bats and num_total is information needed to read the templates. All of these can be set by running 'loading_init'.

Features_key links the row number to a position in the original dictionary and features_freq are the first five columns unscaled.

---

calc_features2(rectangles, regions, templates, list_bats, num_total), return(features)

Variant used in DML code. This function uses non-nested dictionaries.

---

compare_img2(img1, img2), return(score)

Calculates the SSIM between two images and returns it as a score. Images are scaled to the largest image.

---

calc_num_regions(regions), return(num_reg)

Calculates number of regions in a dictionary.

---

cor_plot(features, index, \**optional), return(correlation)

Calculates the correlation between different features. Index is a 2-tuple with the start and stop index. 'Export' can be used to save the plot.

---

**Module 4: AD4_SOM:**

---

fit_SOM(list_files, \**optional), return(net, raw_data)

Fit a self-organising map on a list of audio files. Optional arguments: full, features, dim1, dim2, n_iter (number of iterations), init_learning_rate, normalise_data, normalise_by_column (normalise data per column).

If 'full' is set to True, the list is ignored and all wav files within the folder are used.

If 'features' is given, the audio files aren't analyzed but this matrix is used directly to fit the SOM

---

SOM(raw_data, network_dim, n_iter, init_learning_rate, normalise_data, normalise_by_column, \**optional), return(net)

Subcode called in fit_som. Optional argument 'export'=name can be used to save the map.

---

find_bmu, decay_radius, decay_learning_rate, calculate_influence

Subcodes needed to run SOM.

---

calc_Umat(net), return(U)

Calculates the U-matrix (distance from each neuron to nearest neuron).

---

calc_BMU_scores(data, net), return(score_BMU)

Calculates the best matching unit for each datapoint.

---

calc_net_features(net, \**optional), return(net_features)

Transforms the network to a more suitable form.

---

plot_U(net, \**optional), return()

Calculates and plots the U-matrix. 'Export' can be used to save the plot.

---

heatmap_neurons(M, \**optional), return()

Shows a heatmap of the neurons (number of matches per neuron).

---

calc_output(list_files, net, \**optional), return(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2)

Calculates the features and other output for a list of files and a SOM. If the optional argument 'full' is True all files in the Audio_data folder are analysed. With the optional argument 'Audio_data' an alternate pathway can be specified.

---

rearrange_output(net_label, features, features_key, features_freq, rectangles, regions, spectros, list_files2, net, \**optional), return(full_region, full_rectangle, full_spectro, full_name)

Rearranges the output to group per neuron for the plot function. Optional arguments are 'dim1' and 'dim2' for the SOM (if not given the standard values are read from set_parameters).

---

check_key(regions, temp_key), return(new_key)

Matches indexes rectangles and regions.

---

calc_matching(full_name, \**optional), return(M)

Calculates the number of matches per neuron. (Optional arguments: dim1, dim2)

---

plot_region_neuron(full_region, full_rectangle, full_spectro, full_name, dim1, dim2, point, \**optional), return()

Plots a datapoint matching with a specific neuron. Neuron is specified with dim1 and dim2 and the datapoint with 'point'. Points are ordered from smallest distance to largest (so point 0 is the closest point).

---

calc_context_spec(spectros, k, temp_key, \**optional), return(context_spec, extra_time)

Expands the spectrogram window for plot_region_neuron. The paramters used are: spect_window (size of a window), spect_overlap_window (overlap between windows) and context_window (extra windows added on either side). Returns context_spec (new windows) and extra_time (time added in total, needed to label x axis).

---

calc_FI_matrix(spectros), return(FI_matrix)

Transforms a region into an intensity-frequency matrix.

---

calc_maxc(full_names, \**optional), return(max_c)

Calculates the maximum amount of matches. Needed to set a limit on the interactive plot slider.

---

fit_dml(\**optional), return(D)

Fits a distance metric learning matrix. Optional argument 'path' can be used to specify a different pathway to the templates. Optional arguments 'data_X' and 'data_Y' can be used to fit a dml on custom data. 'export' is used to export the matrix.

---

**Module 5: AD5_MDS:**

---

calc_dist_matrix(net_features, Axis, \**optional), return(D)

Calculates the distance between rows (axis=0) or colums (axis=1). If the optional argument 'raw_data' is given, it is concatenated with net_features.

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

plot_MDS2(pos, dim1, dim2), return()

Plots an MDS of data and SOM neurons (dim1 and dim2 are dimensions of the SOM). Neurons are red and datapoints are black.

---
