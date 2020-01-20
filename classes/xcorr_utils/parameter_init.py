import ConfigParser

config = ConfigParser.ConfigParser()
config.read("./config.cfg")


#[DATASET]
dataset_path = config.get("DATASET", "dataset_path")
components = config.get("DATASET", "components").split(",")
years = config.get("DATASET", "years").split(",")
build_dataset = config.getboolean("DATASET", "build_dataset")
dataset_name = config.get("DATASET", "dataset_name")
file_type = config.get("DATASET", "file_type")

#[XCORR-ACORR]
maxlag = config.getint("XCORR", "maxlag")
max_waveforms = config.getint("XCORR", "max_waveforms")
input_path = config.get("XCORR", "input_path")
number_of_cpus = config.getint("XCORR", "number_of_cpus")
verbose = config.getboolean("XCORR", "verbose")
save_path = config.get("XCORR", "save_path")
extended_save = config.getboolean("XCORR", "extended_save")

#[TIMEDOMAIN-NORMALIZATION]
binary_normalization = config.getboolean("TIMEDOMAIN-NORMALIZATION", "binary_normalization")
running_absolute_mean_normalization = config.getboolean("TIMEDOMAIN-NORMALIZATION", "running_absolute_mean_normalization")
filter_num = config.getint("TIMEDOMAIN-NORMALIZATION", "filters")
filters = []
i = 1
while i <= filter_num:
    f = map(float, config.get("TIMEDOMAIN-NORMALIZATION", "filter%s" % (i)).split(','))
    filters.append(f)
    i += 1
envsmooth = config.getint("TIMEDOMAIN-NORMALIZATION", "envsmooth")
env_exp = config.getfloat("TIMEDOMAIN-NORMALIZATION", "env_exp")
min_weight = config.getfloat("TIMEDOMAIN-NORMALIZATION", "min_weight")
taper_length_timedomain = config.getint("TIMEDOMAIN-NORMALIZATION", "taper_length_timedomain")
apply_broadband_filter_tdn = config.getboolean("TIMEDOMAIN-NORMALIZATION", "apply_broadband_filter_tdn")
broadband_filter_tdn = map(float, config.get("TIMEDOMAIN-NORMALIZATION", "broadband_filter_tdn").split(","))
filter_order_tdn = config.getint("TIMEDOMAIN-NORMALIZATION", "filter_order_tdn")
plot = config.getboolean("TIMEDOMAIN-NORMALIZATION", "plot")

#[SPECTRAL-WHITENING]
apply_spectral_whitening = config.getboolean("SPECTRAL-WHITENING", "apply_spectral_whitening")
spectrumexp = config.getfloat("SPECTRAL-WHITENING", "spectrumexp")
espwhitening = config.getfloat("SPECTRAL-WHITENING", "espwhitening")
taper_length_whitening = config.getint("SPECTRAL-WHITENING", "taper_length_whitening")
apply_broadband_filter_whitening = config.getboolean("SPECTRAL-WHITENING", "apply_broadband_filter_whitening")
broadband_filter_whitening = map(float, config.get("SPECTRAL-WHITENING", "broadband_filter_whitening").split(","))
filter_order_whitening = config.getint("SPECTRAL-WHITENING", "filter_order_whitening")