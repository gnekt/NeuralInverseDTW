from torch import full 
import yaml

config = yaml.safe_load(open("Configs/config.yml"))
MEL_PARAMS = config.get('preprocess_params', {"n_mel_band":80})

pad_mel_band_value = .0 # <pad> Float
bos_mel_band_value = .0 # <bos> Float
eos_mel_band_value = .0 # <eos> Float

pad_token = full((1,MEL_PARAMS["spect_params"]["n_mel_band"]),pad_mel_band_value)
bos_token = full((1,MEL_PARAMS["spect_params"]["n_mel_band"]),bos_mel_band_value)
eos_token = full((1,MEL_PARAMS["spect_params"]["n_mel_band"]),eos_mel_band_value) 

#############################################3
# External token used are <pad>, <bos>, <eos>
# each of them have a custom vector representation
# think alway that in this case the "embedding" are the frequencies themselves
# so in this case <pad> is a t_frame having in all the frequencies a value in energy eq to 997