import ConfigParser

config = ConfigParser.ConfigParser()
config.read("./config.cfg")


#[DATASET]
dataset_path = config.get("DATASET", "dataset_path")
components = config.get("DATASET", "components").split(",")
years = config.get("DATASET", "years").split(",")
build_dataset = config.getboolean("DATASET", "build_dataset")
dataset_name = config.get("DATASET", "dataset_name")

#[XCORR]
maxlag = config.getint("XCORR", "maxlag")
input_path = config.get("XCORR", "input_path")
number_of_cpus = config.getint("XCORR", "number_of_cpus")
verbose = config.getboolean("XCORR", "verbose")
save_path = config.get("XCORR", "save_path")
extended_save = config.getboolean("XCORR", "extended_save")
file_type = config.get("XCORR", "file_type")

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
taper_lenght_timedomain = config.getint("TIMEDOMAIN-NORMALIZATION", "taper_lenght_timedomain")
broadband_filter = map(float, config.get("TIMEDOMAIN-NORMALIZATION", "broadband_filter").split(","))
plot = config.getboolean("TIMEDOMAIN-NORMALIZATION", "plot")

#[SPECTRAL-WHITENING]
spectrumexp = config.getfloat("SPECTRAL-WHITENING", "spectrumexp")
espwhitening = config.getfloat("SPECTRAL-WHITENING", "espwhitening")
taper_length_whitening = config.getint("SPECTRAL-WHITENING", "taper_length_whitening")
