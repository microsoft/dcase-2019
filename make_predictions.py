import csv
import os
import re

import numpy as np
from sklearn.externals import joblib
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# process label names
label_names = joblib.load('label_order.pkl')
label_names = [re.sub('_presence', '', label_names[i]) for i in range(len(label_names))]


def reorder_labels_for_submission(list_or_array):
    if type(list_or_array) == torch.Tensor:
        list_or_array = list(list_or_array.to('cpu').numpy())
    list_or_array = list(list_or_array)
    return list_or_array[8:] + list_or_array[0:8]

def make_prediction_csv(model, prediction_path, embed=True, mode='all', test_path=''):
    """
    Writes CSV file and evaluates for AUPRC scores
    :param model: pytorch model
    :param prediction_path: path to save .csv file of predictions
    :param embed: True if using vgg embeddings; else False
    :param mode: 'all', 'coarse', 'fine'
    :param test_path: directory of validate pickle files
    """

    with open(prediction_path, 'w') as c:
        writer = csv.writer(c, delimiter=',')

        header = ['audio_filename'] + reorder_labels_for_submission(label_names)
        writer.writerow(header)
        basepath = test_path
        data_rows = []
        for filename in os.listdir(basepath):
            audio_filename = filename[0:9] + '.wav'
            if embed:
                spectrogram, vgg, label = joblib.load(os.path.join(basepath, filename))
                spectrogram = np.expand_dims(spectrogram, axis=0)
                spectrogram = np.expand_dims(spectrogram, axis=0)

                vgg = torch.from_numpy(vgg.flatten().reshape((1, 1280)))
                spectrogram = spectrogram.astype(np.float32)
                spectrogram = torch.from_numpy(spectrogram)

                in_data = (spectrogram.to(device), vgg.to(device))  # emb
            else:
                spectrogram, label = joblib.load(os.path.join(basepath, filename))
                spectrogram = np.expand_dims(spectrogram, axis=0)
                spectrogram = np.expand_dims(spectrogram, axis=0)
                spectrogram = spectrogram.astype(np.float32)
                spectrogram = torch.from_numpy(spectrogram)
                in_data = (spectrogram.to(device))  # emb
            with torch.no_grad():
                results = model(in_data)
                results = torch.sigmoid(results[0])

                results = results.to('cpu').detach().numpy()
                if mode == 'fine':
                    coarse_labels = []
                    fine_label_names = label_names[8:]
                    coarse_label_dict = {i: [] for i in range(8)}
                    for i, r in enumerate(results):
                        coarse_label_dict[int(fine_label_names[i][0]) - 1].append(r)
                    for i in range(8):
                        coarse_labels.append(max(coarse_label_dict[i]))
                    results = coarse_labels + list(results)
                elif mode == 'coarse':
                    results = list(results) + [0 for i in range(29)]
                results = reorder_labels_for_submission(results)
                data_rows.append([audio_filename] + results)
        writer.writerows(data_rows)