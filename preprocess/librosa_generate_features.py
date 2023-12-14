import librosa
import librosa.display
import numpy as np


def extract_features_librosa(audio_path):
    try:

        features_data = []
        if True:
            y, sr = librosa.load(audio_path, sr=16000)

            # 提取音频特征
            stft = np.abs(librosa.stft(y))

            # fmin 和 fmax 对应于人类语音的最小最大基本频率
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, S=stft, fmin=70, fmax=400)
            pitch = []
            for j in range(magnitudes.shape[1]):
                index = magnitudes[:, j].argmax()
                pitch.append(pitches[index, j])

            pitch_tuning_offset = librosa.pitch_tuning(pitches)
            pitchmean = np.mean(pitch)
            pitchstd = np.std(pitch)
            pitchmax = np.max(pitch)
            pitchmin = np.min(pitch)

            # 频谱质心
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            cent = cent / np.sum(cent)
            meancent = np.mean(cent)
            stdcent = np.std(cent)
            maxcent = np.max(cent)

            # 谱平面
            flatness = np.mean(librosa.feature.spectral_flatness(y=y))

            # 使用系数为50的MFCC特征
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50).T, axis=0)
            mfccsstd = np.std(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50).T, axis=0)
            mfccmax = np.max(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=50).T, axis=0)

            # 色谱图
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

            # 梅尔频率
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

            # ottava对比
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)

            # 过零率
            zerocr = np.mean(librosa.feature.zero_crossing_rate(y))

            S, phase = librosa.magphase(stft)
            meanMagnitude = np.mean(S)
            stdMagnitude = np.std(S)
            maxMagnitude = np.max(S)

            # 均方根能量
            rmse = librosa.feature.rms(S=S)[0]
            meanrms = np.mean(rmse)
            stdrms = np.std(rmse)
            maxrms = np.max(rmse)

            features = np.array([
                flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
                maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
                pitch_tuning_offset, meanrms, maxrms, stdrms
            ])

            features = np.concatenate((features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast)).astype(np.float32)
            # 将特征保存到列表
            features_data.append(features)


        return features
    except Exception as e:
        print("-----error-----", str(e))

