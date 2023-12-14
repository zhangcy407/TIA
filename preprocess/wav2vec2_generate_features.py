import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import logging

logging.getLogger("transformers").setLevel(logging.CRITICAL)
device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_features_wav2vec2(audio_path):
    try:
        if True:

            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('../wav2vec2-base-960h')
            model = Wav2Vec2Model.from_pretrained('../wav2vec2-base-960h')
            model = model.to(device)

            waveform, sample_rate = torchaudio.load(audio_path)
            waveform_16k = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
            sample_rate = 16000

            input_values = feature_extractor(waveform_16k.squeeze().numpy(),
                                             return_tensors="pt", sampling_rate=sample_rate).input_values
            input_values = input_values.to(device)

            with torch.no_grad():
                outputs = model(input_values)
                hidden_states = outputs.last_hidden_state[-1, :, :]
                hidden_states = hidden_states.to("cpu")
                hidden_states_array = hidden_states.numpy()
        return hidden_states_array
    except Exception as e:
        print("-----error-----", str(e))
