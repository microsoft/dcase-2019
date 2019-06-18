from collections import defaultdict
import os

import joblib
import numpy as np
import pandas as pd
#from sklearn.externals import joblib
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## coarse_number: (start_index, end_index) for entire 37 labels (with 0:8 being coarse labels, 8: being fine)
label_hierarchy = {
    1: (8, 12),
    2: (12, 17),
    3: (17, 18),
    4: (18, 22),
    5: (22, 27),
    6: (27, 31),
    7: (31, 36),
    8: (36, 37)
}

# label_dict = joblib.load('label_order.pkl')


class AudioDataset(Dataset):
    def __init__(self, data_path, with_embeddings=True):
        if not os.path.exists(data_path):
            raise Exception('data path does not exist')
        self.data_path = [
            os.path.join(data_path, f) for f in os.listdir(data_path)
        ]
        self.data_len = len(self.data_path)
        self.with_embeddings = with_embeddings

    def __getitem__(self, index):

        self.filename = os.path.basename(self.data_path[index])
        desired_spectrogram_shape = (1, 128, 862)

        if not self.with_embeddings:
            spectrogram, label = joblib.load(self.data_path[index])
            spectrogram = np.expand_dims(spectrogram, 0)
            if spectrogram.shape != desired_spectrogram_shape:
                zero_pad = np.zeros((1, 128, 2))
                spectrogram = np.concatenate((spectrogram, zero_pad), axis=2)
            spectrogram = spectrogram.astype(np.float32)
            spectrogram = torch.from_numpy(spectrogram)
            spectrogram = spectrogram.to(device)
            label = label.astype(np.float32)
            label = torch.from_numpy(label)[
                0:8]  # [0:8] for coarse-only  #[8:] for fine-only
            label = label.to(device)
            return spectrogram, label

        # spectrogram, l3_emb, vgg_emb, label = joblib.load(self.data_path[index])
        spectrogram, vgg_emb, label = joblib.load(self.data_path[index])
        spectrogram = np.expand_dims(spectrogram, 0)
        # l3_emb = l3_emb.reshape((1, 256, 192))

        # add zeros to files that were short 2 frames
        if spectrogram.shape != desired_spectrogram_shape:
            zero_pad = np.zeros((1, 128, 2))
            spectrogram = np.concatenate((spectrogram, zero_pad), axis=2)
        spectrogram = spectrogram.astype(np.float32)
        spectrogram = torch.from_numpy(spectrogram)
        # l3_emb = torch.from_numpy(l3_emb)
        vgg_emb = torch.from_numpy(vgg_emb.flatten())
        label = label.astype(np.float32)
        label = torch.from_numpy(label)[
            0:8]  # [0:8] for coarse-only  #[8:] for fine-only

        spectrogram = spectrogram.to(device)
        # l3_emb = l3_emb.to(device)
        vgg_emb = vgg_emb.to(device)
        label = label.to(device)

        return (spectrogram,
                vgg_emb), label  #(spectrogram, l3_emb, vgg_emb), label

    def __len__(self):
        return self.data_len


def get_hierarchy_files(index_to_files_dict, coarse_label_index):
    return index_to_files_dict[coarse_label_index]


class AudioDatasetFine(Dataset):
    def __init__(self, data_path, coarse_label_index, index_to_files_dict):
        if not os.path.exists(data_path):
            raise Exception('data path does not exist')
        self.base_path = data_path
        self.data_path = get_hierarchy_files(index_to_files_dict,
                                             coarse_label_index)

        # TODO: option to assign all extra files to a negative class!


        self.data_len = len(self.data_path)
        self.coarse_label_index = coarse_label_index

        print(f'Data len: {self.data_len}')

    def __getitem__(self, index):

        self.filename = os.path.basename(self.data_path[index])
        desired_shape = (1, 128, 862)
        # spectrogram, l3_emb, vgg_emb, label = joblib.load(self.data_path[index])
        spectrogram, vgg_emb, label = joblib.load(
            os.path.join(self.base_path, self.data_path[index]))
        label_start, label_end = label_hierarchy[self.coarse_label_index + 1]
        label = label[label_start:label_end]
        spectrogram = np.expand_dims(spectrogram, 0)
        # l3_emb = l3_emb.reshape((1, 256, 192))

        # add zeros to files that were short 2 frames
        if spectrogram.shape != desired_shape:
            zero_pad = np.zeros((1, 128, 2))
            spectrogram = np.concatenate((spectrogram, zero_pad), axis=2)
        spectrogram = spectrogram.astype(np.float32)
        spectrogram = torch.from_numpy(spectrogram)
        # l3_emb = torch.from_numpy(l3_emb)
        vgg_emb = torch.from_numpy(vgg_emb.flatten())
        label = label.astype(np.float32)
        label = torch.from_numpy(
            label)  # [0:8] for coarse-only  #[8:] for fine-only

        spectrogram = spectrogram.to(device)
        # l3_emb = l3_emb.to(device)
        vgg_emb = vgg_emb.to(device)
        label = label.to(device)

        return (spectrogram, vgg_emb), label  #l3_emb,

    def __len__(self):
        return self.data_len


# TODO: Deprecated. Use a precomputed index_to_files_dict instead, as in get_hierarchy_files above.
def load_dataset_from_path(path, coarse_label_index):
    print(
        'Warning: load_dataset_from_path is deprecated. Use precomputed index_to_files_dict instead.'
    )
    all_files = [os.path.join(path, f) for f in os.listdir(path)]
    X = []
    Y = []
    desired_shape = (1, 128, 862)
    for f in all_files:
        spectrogram, l3_emb, vgg_emb, label = joblib.load(f)
        if label[coarse_label_index] == 1:
            spectrogram = np.expand_dims(spectrogram, 0)
            l3_emb = l3_emb.reshape((1, 256, 192))
            label_start, label_end = label_hierarchy[coarse_label_index + 1]
            label = label[label_start:label_end]
            # add zeros to files that were short 2 frames
            if spectrogram.shape != desired_shape:
                zero_pad = np.zeros((1, 128, 2))
                spectrogram = np.concatenate((spectrogram, zero_pad), axis=2)
            spectrogram = spectrogram.astype(np.float32)
            l3_emb = l3_emb.astype(np.float32)
            vgg_emb = vgg_emb.astype(np.float32)
            label = label.astype(np.float32)

            spectrogram = torch.from_numpy(spectrogram)
            l3_emb = torch.from_numpy(l3_emb)
            vgg_emb = torch.from_numpy(vgg_emb.flatten())
            label = torch.from_numpy(label)

            spectrogram = spectrogram.to(device)
            l3_emb = l3_emb.to(device)
            vgg_emb = vgg_emb.to(device)
            label = label.to(device)

            X.append((spectrogram, l3_emb, vgg_emb))
            Y.append(label)

    return X, Y


TRAIN_WEIGHTS = [
    3.29014598540146, 6.733552631578948, 46.97959183673469, 20.76851851851852,
    6.633116883116883, 51.24444444444445, 3.817622950819672, 21.17924528301887,
    122.73684210526316, 20.568807339449542, 11.988950276243093,
    155.73333333333332, 234.1, 35.734375, 179.84615384615384,
    390.8333333333333, 101.21739130434783, 46.97959183673469,
    96.95833333333333, 260.22222222222223, 292.875, 782.6666666666666,
    17.511811023622048, 586.75, 28.759493670886076, 28.02469135802469,
    260.22222222222223, 137.2941176470588, 0, 0, 292.875, 4.6650602409638555,
    137.2941176470588, 1174.5, 0, 586.75, 21.17924528301887
]

TEST_WEIGHTS = [
    1.248730964467005, 12.424242424242424, 54.375, 25.058823529411764, 4.5375,
    19.136363636363637, 1.4748603351955307, 72.83333333333333, 442.0,
    11.305555555555555, 4.753246753246753, 0, 0, 220.5, 0, 0, 0, 54.375, 220.5,
    442.0, 442.0, 442.0, 16.03846153846154, 0, 8.844444444444445, 43.3, 0,
    39.27272727272727, 0, 442.0, 0, 1.9731543624161074, 43.3, 0, 0, 0,
    72.83333333333333
]
