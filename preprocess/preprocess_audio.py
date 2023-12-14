from librosa_generate_features import extract_features_librosa
from wav2vec2_generate_features import extract_features_wav2vec2
from sklearn.model_selection import train_test_split
import random
import pickle
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def preprocess_audio(audio_folder_pos, audio_folder_neg, feature_path, split_ratio=(0.7, 0.2, 0.1)):
    # List all audio files
    audio_files_pos = list(Path(audio_folder_pos).glob("*.wav"))
    audio_files_neg = list(Path(audio_folder_neg).glob("*.wav"))

    # Assign labels
    labeled_data = [(str(file), 1) for file in audio_files_pos] + [(str(file), 0) for file in audio_files_neg]

    # Shuffle data
    random.seed(3407)
    random.shuffle(labeled_data)

    # Split into training, validation, and test sets
    train_val_data, test_data = train_test_split(labeled_data, test_size=split_ratio[2], random_state=3407)
    train_data, val_data = train_test_split(train_val_data,
                                            test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
                                            random_state=3407)

    # Function to process and save data
    def process_and_save(data, set_type):
        librosa_features = []
        wav2vec2_features = []
        labels = []
        i = 0
        with tqdm(total=len(data), desc=(set_type + ' features'), unit='') as tqdm_per:
            for file, label in data:
                librosa_feature = extract_features_librosa(file)
                wav2vec2_feature = extract_features_wav2vec2(file)
                librosa_features.append(librosa_feature)
                wav2vec2_features.append(wav2vec2_feature)
                labels.append(label)
                i += 1

                tqdm_per.update()
            tqdm_per.close()
        librosa_dataset = {'features': librosa_features, 'labels': labels}
        wav2vec2_dataset = {'features': wav2vec2_features, 'labels': labels}
        # Save as pkl file
        with open(feature_path / 'wav2vec2' / f"{set_type}_data.pkl", 'wb') as file:
            pickle.dump(librosa_dataset, file)
        with open(feature_path / 'librosa' / f"{set_type}_data.pkl", 'wb') as file:
            pickle.dump(wav2vec2_dataset, file)

    # Process and save training, validation, and test data
    process_and_save(train_data, 'train')
    print("--train features extra acccess--")
    process_and_save(val_data, 'val')
    print("--val features extra acccess--")
    process_and_save(test_data, 'test')
    print("--test features extra acccess--")

    return {'train': train_data, 'val': val_data, 'test': test_data}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio data")
    parser.add_argument('--positive_folder', type=str, default='../TIAaudio/yes',
                        help="Path to folder containing positive audio files")
    parser.add_argument('--negative_folder', type=str, default='../TIAaudio/no',
                        help="Path to folder containing negative audio files")
    parser.add_argument('--output_path', type=str, default='../TIAaudio/datasets/')
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    preprocess_audio(args.positive_folder, args.negative_folder, output_path)
