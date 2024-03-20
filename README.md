# TIA: A Teaching Intonation Assessment Dataset in Real Teaching Situations

This work is accepted by ICASSP 2024.

## Dataset

You can download the TIA dataset and features from BaiduNetDisk.
Link:https://pan.baidu.com/s/1sMdroZ6lcU0JaC20ShQmJg?pwd=iuss password:iuss

## Environment

```
python==3.8.0
pytorch==1.12.1
```

## Usage

+ For Training TIAM on TIA:

```bash
python run.py --model TIAM
```

## Feature extraction and Pretrained Model Preparation

Feature extraction file is in `./preprocess/preprocess_audio.py`

The feature we used is from wav2vec2, which could be downloaded from https://huggingface.co/facebook/wav2vec2-base-960h.

## Citation

S. Liu, C. Zhang, B. Li, N. Qin, H. Cheng and H. Zhang, "TIA: A Teaching Intonation Assessment Dataset in Real Teaching Situations," ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 1066-1070, doi: 10.1109/ICASSP48485.2024.10447407.