A list of all optional arguments used.

Audio_data: specify pathway to alternate folder for the audio_data. Used in functions: AD4.make_list, AD4.fit_SOM, AD4.calc_output

folder: specify a folder within Audio_data. Used in functions: AD4.make_list, AD4.fit_SOM, AD4.calc_output, AD4.evaluation_SOM

subfolders: if set to True, subfolders within the audio_data are also read.  Used in functions: AD4.make_list, AD4.fit_SOM, AD4.calc_output, AD4.evaluation_SOM

full: if set to True, all data within the audio_data folder is read. Used in functions: AD4.make_list, AD4.fit_SOM, AD4.calc_output, AD4.evaluation_SOM

Templates: pathway to a folder that stores the templates. Used in functions: AD1.read_templates, AD1.loading_init, AD1.set_batscolor, AD1.print_features, AD1.set_numbats, AD2.create_template, AD2.create_template, AD4.fit_SOM, AD4.calc_output, AD4.fit_dml

template_type: type of template ('regular', 'dml' or 'evaluate'). Used in functions: AD2.create_template

DML: use a DML-matrix. If not given, the program defaults to the identity matrix. Used in functions: AD4.fit_SOM, AD4.SOM, AD4.calc_output, AD4.calc_BMU_scores, AD4.evaluation_SOM

path: alternate pathway to load data (dml, som,...). Used in functions: AD1.import_map, AD1.import_dml, AD4.evaluation_SOM, AD4.print_evaluate, AD4.print_evaluate2, AD4.calc_total_bats, AD4.KNN_calc, AD4.calc_PE

spect_window: manually overwrite the size of the window. Used in functions: AD2.spect, AD2.spect_loop, AD2.show_region, AD2.show_mregions, AD2.create_template, AD4.fit_SOM, AD4.calc_output, AD4.calc_context_spec, AD4.rearrange_output

spect_window_overlap: manually overwrite the overlap between windows. Used in functions: AD2.spect, AD2.spect_loop, AD2.create_template, AD4.fit_SOM, AD4.calc_output, AD4.calc_context_spec, AD4.rearrange_output

sr: sampling rate. Used if the program can't load because of metadata. Librosa will be used instead of scipy. Used in functions: AD2.spect, AD2.spect_loop, AD2.create_template, AD4.fit_SOM, AD4.calc_output

channel: 'l' or 'r' for stereo files. Used in functions: AD2.spect, AD2.spect_loop, AD2.create_template, AD4.fit_SOM, AD4.calc_output

min_spec_freq: manually overwrite the minimum frequency of the window. Used in functions: AD2.spect_plot, AD2.show_region, AD2.show_mregions, AD2.spect_loop, AD4.fit_SOM, AD4.calc_output

max_spec_freq: manually overwrite the maximum frequency of the window. Used in functions: AD2.spect_plot, AD2.show_region, AD2.show_mregions, AD2.spect_loop, AD4.fit_SOM, AD4.calc_output

nosub: turn off the subtraction (if set to True). This manually overwrite the minimum frequency of the window. Used in functions: AD2.spect_plot, AD2.spect_loop, AD4.fit_SOM, AD4.calc_output

X: manually overwrite the threshold to find ROIs. Used in functions: AD2.spect_loop, AD2.create_template, AD4.fit_SOM, AD4.calc_output

kern: manually overwrite the kernel size to find ROIs. Used in functions: AD2.spect_loop, AD2.create_template, AD4.fit_SOM, AD4.calc_output

exp_factor: expansion factor for TE-data. Used in functions: AD2.spect_loop, AD2.create_template, AD4.fit_SOM, AD4.calc_output

max_roi: manually overwrite the maximum number of ROIs in a single spectrogram. Used in functions: AD2.overload, AD2.spect_loop, AD2.create_template, AD4.fit_SOM, AD4.calc_output

export: export a figure. Set to a pathway or name. Extension is added automatically. Used in functions: AD4.cor_plot, AD4.SOM, AD4.fit_SOM, AD4.plot_U, AD4.heatmap_neurons, AD4.fit_dml

dim1: manually overwrite default first dimension of a SOM. If dim1 is given, dim2 needs to be given as well. Used in functions: AD4.fit_SOM, AD4.calc_net_features, AD4.rearrange_output, AD4.calc_matching, AD4.calc_maxc, AD4.evaluation_SOM

dim2: manually overwrite default second dimension of a SOM. If dim2 is given, dim1 needs to be given as well. Used in functions: AD4.fit_SOM, AD4.calc_net_features, AD4.rearrange_output, AD4.calc_matching, AD4.calc_maxc, AD4.evaluation_SOM

n_iter: manually overwrite default number of iterations of a SOM. Used in functions: AD4.fit_SOM,

init_learning_rate: manually overwrite default initial learning rate of a SOM. Used in functions: AD4.fit_SOM,

normalise_data: if set to True, data is normalised before a SOM. Set to False by default because a DML is used. Used in functions: AD4.fit_SOM,

normalise_by_column: if set to True, data is normalised per column before a SOM. Set to False by default because a DML is used. Used in functions: AD4.fit_SOM,

features: give a set of features to a SOM instead of calculating them from a set of files. Overwrites the list of files given. Used in functions: AD4.fit_SOM,

init_learning_rate: manually overwrite default initial learning rate of a SOM. Used in functions: AD4.fit_SOM,

context_window_freq: manually overwrites the context window frequency padding. Used in functions: AD4.plot_region_neuron

context_window: manually overwrites the number of context windows. Used in functions: AD4.calc_context_spec, AD4.rearrange_output

data_X: manually load in features to fit a DML. If data_X is given, data_Y needs to be given as well.. Used in functions: AD4.fit_dml

data_X: manually load output data to fit a DML. If data_Y is given, data_X needs to be given as well. Used in functions: AD4.fit_dml

templates_features: read a different set of features for the evaluation of a SOM. Used in functions: AD4.evaluation_SOM

SOM: give a SOM-map for the evaluation of a SOM. Used in functions: AD4.evaluation_SOM

Title: set a title for the evaluation SOM plot. Used in functions: AD4.evaluation_SOM

Plot_flag: if set to True, a plot is made for the evaluation of a SOM. If set to False, the plot is skipped. Used in functions: AD4.evaluation_SOM

fit_eval: if True, use eval data on SOM evaluation. Used in functions: AD4.evaluation_SOM

List_files: use a list of files to fit a SOM for the SOM evaluation. Used in functions: AD4.evaluation_SOM

K: manually overwrite the default K of 3 for a KNN. Used in functions: AD4.KNN_calc

raw_data: extra data used to calculate a distance matrix. Used in functions: AD5.calc_dist_matrix
