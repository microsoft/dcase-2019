from collections import defaultdict
import os

import librosa
import numpy as np
import pandas as pd
import yaml
from sklearn.externals import joblib

from audio_augmentation import save_augmented_files, make_log_mel_spectrogram

def _choose_label(list_of_labels):
    # given an array of choices from annotators, return one label
    valid_labels = [n for n in list_of_labels if n != -1]
    yes_labels = [n for n in valid_labels if n == 1]
    return 1 if valid_labels and float(len(yes_labels)) / len(valid_labels) > .5 else 0

def get_labels(taxonomy_yaml_path, annotations_path):
    # load CSV and YAML files
    val_labels = pd.read_csv(annotations_path)
    with open(taxonomy_yaml_path, 'r') as yaml_file:
        label_taxonomy = yaml.load(yaml_file)

    # flatten hierarchy of labels
    label_order = []  # list of label names in flattened order #TODO: write these to file for ease of access to \
    # label names from k-hot labels
    label_dict = {}
    label_number = 0
    # TODO: make these loops work with different levels of hierarchies (e.g. Google AudioSet)
    for k, v in label_taxonomy['coarse'].items():
        if v not in label_dict.keys():
            label_name = '{}_{}_presence'.format(k, v)
            label_dict[label_name] = label_number
            label_order.append(label_name)
            label_number += 1
    for k, v in label_taxonomy['fine'].items():
        for kk, vv in v.items():
            if vv not in label_dict.keys():
                label_name = '{}-{}_{}_presence'.format(k, kk, vv)
                label_dict[label_name] = label_number
                label_order.append(label_name)
                label_number += 1

    # process multiple annotations per file to one label array per file
    filename_to_label_dict = defaultdict(list)  # file_name: [[annotation_1], [annotation_2], ...]
    label_dict_final = {}  # file_name: [k-hot label]
    for i, r in val_labels.iterrows():
        filename_to_label_dict[r.audio_filename].append([r[label_name] for label_name in label_order])

    for k, v in filename_to_label_dict.items():
        label_array = np.zeros(len(label_order), dtype=np.int)
        label_choices = np.array(v, dtype=np.int)
        for i in range(len(label_array)):
            label_choice = label_choices[:, i]
            label_array[i] = _choose_label(label_choice)
        label_dict_final[k] = label_array

    return label_dict_final


if __name__ == '__main__':

    audio_dir = '/path/to/audio/files'
    output_dir = '/path/to/save/pickle/files'
    taxonomy_yaml_path = './dcase_files/dcase-ust-taxonomy.yaml'
    annotations_path = './dcase_files/annotations.csv'
    sample_rate = 44100
    augment_files = True


    filename_to_label = get_labels(taxonomy_yaml_path, annotations_path)
    for file in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file)
        label = filename_to_label[file]
        if augment_files:
            save_augmented_files(file_path, label, output_dir, volume=True, pitch=True, bkgrd=True, sr=sample_rate)
        else:
            y, sr = librosa.load(file_path, sr=sample_rate)
            S = make_log_mel_spectrogram(y, sr)
            save_name = file.replace('.wav', '.pkl')
            joblib.dump((S, label), os.path.join(output_dir, save_name))
