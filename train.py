"""trains model, saves checkpoint files and csv files of best metrics"""

from datetime import datetime
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_dataset import AudioDataset, TRAIN_WEIGHTS
from make_predictions import make_prediction_csv
from metrics import evaluate, micro_averaged_auprc

TRAIN_MODE = 'all'  # options: all (37 classes), coarse (8 classes), fine (29 classes)
train_mode_dict = {'all': 37, 'coarse': 8, 'fine': 29}  # train_mode: num_labels

DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

RUN_NAME = DATE
PREDICTION_PATH = f'csvs/{RUN_NAME}.csv'
ANNOTATIONS_PATH = "dcase_files/annotations.csv"
YAML_PATH = "dcase_files/dcase-ust-taxonomy.yaml"

BATCH_SIZE = 32
NUM_CLASSES = train_mode_dict[TRAIN_MODE]
NUM_EPOCHS = 10000

## load model from checkpoint
CHECKPOINT = True
CHECKPOINT_PATH = "models/20190610_083507_coarse=0.777_fine=0.644.ckpt"

## apply weights to classes to deal with unbalanced dataset
APPLY_CLASS_WEIGHTS = False
WEIGHT_SMOOTHING = .1  # 1: balanced weight, < 1: reduced weight delta, > 1: increased weight delta

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = "path/to/train/dataset/pickle/files.pkl"
test_dir = "path/to/validate/dataset/pickle/files.pkl"

TRAIN = AudioDataset(train_dir, with_embeddings=True)
TEST = AudioDataset(test_dir, with_embeddings=True)

TRAIN_LOADER = DataLoader(dataset=TRAIN, batch_size=BATCH_SIZE, shuffle=True)
TEST_LOADER = DataLoader(dataset=TEST, batch_size=BATCH_SIZE, shuffle=True)


class ConvBlock(nn.Module):
    """creates a convolutional layer with optional maxpool, batchnorm, and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 batchnorm=True, maxpool=True, maxpool_size=(2, 2), dropout=None):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding)

        if maxpool:
            self.mp = nn.MaxPool2d(maxpool_size, stride=maxpool_size)
        else:
            self.mp = None

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # self.init_weights()

    def forward(self, nn_input):

        x = nn_input

        if self.bn:
            x = F.relu(self.bn(self.conv(x)))
        else:
            x = F.relu(self.conv(x))
        if self.mp:
            x = self.mp(x)
        if self.dropout:
            x = self.dropout(x)

        return x


class AudioCNN(nn.Module):
    """Convolutions over spectrogram; merges with VGG-ish embeddings for fully-connected layers"""
    def __init__(self):
        super(AudioCNN, self).__init__()

        DROPOUT = .5
        self.emb_size = 49152

        # spectrogram convolutions
        self.conv_block_1 = ConvBlock(in_channels=1, out_channels=8, kernel_size=(1, 1), stride=(1, 1),
                                       padding=(0, 0), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=DROPOUT)

        self.conv_block_2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=DROPOUT)

        self.conv_block_3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=(16, 128), stride=(4, 16),
                                       padding=(8, 16), batchnorm=True, maxpool=True, maxpool_size=(4, 4),
                                      dropout=DROPOUT)

        self.conv_block_4 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=DROPOUT)

        self.conv_block_5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=DROPOUT)

        self.conv_block_6 = ConvBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None,
                                      dropout=DROPOUT)

        ## combine output of conv_block_6 with VGG-ish embedding
        self.fc1 = nn.Bilinear(256, 1280, 512, bias=True)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc_final = nn.Linear(256, NUM_CLASSES, bias=True)

        self.fc_dropout = nn.Dropout(.2)

    def forward(self, nn_input):

        x, vgg = nn_input  # (spectrogram, vggish)

        # spectrogram convolutions
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)

        # flatten conv layer and vgg-ish embeddings for fc layers
        x = x.view(x.size(0), -1)
        vgg = vgg.view(vgg.size(0), -1)

        ## fully-connected layers
        x = self.fc1(x, vgg)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        x = self.fc_final(x)
        output = x

        return output


if __name__ == '__main__':

    model = AudioCNN().to(device)

    ## if training from checkpoint, ensure checkpoint matches model class architecture
    if CHECKPOINT:
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint)

    ## Loss function
    if WEIGHT_SMOOTHING:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(np.array(TRAIN_WEIGHTS).astype(np.float32)**WEIGHT_SMOOTHING).to(device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Train the model
    starting_lr = .01 #.001
    lr = starting_lr
    min_lr = 1e-6
    stagnation = 0
    stagnation_threshold = 10
    reduce_lr_rate = .1
    running_loss = 0
    train_losses, test_losses = [], []
    best_micro_auprc_coarse = 0
    best_micro_auprc_fine = 0

    # TODO: print epoch, train_loss, val_loss, micro_auprc, etc.
    optimizer = torch.optim.Adam(model.parameters(), lr=starting_lr)
    for epoch in range(NUM_EPOCHS):
        epoch_losses = []
        if stagnation > stagnation_threshold:
            if lr <= min_lr:
                lr = starting_lr
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                reduce_lr_rate += .1
                print('.' * 50)
                print('reset learning rate to', lr)
                print('.' * 50)
                stagnation = 0
            else:
                lr = lr * reduce_lr_rate
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                print('.' * 50)
                print('reduced learning rate to', lr)
                print('.' * 50)
                stagnation = 0
        for i, (spectrogram, label) in enumerate(TRAIN_LOADER):
            # Forward pass
            outputs = model(spectrogram)
            loss = criterion(outputs, label)
            epoch_losses.append(loss.item())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:', epoch)
        print('lr:', lr)
        print('Train loss:', np.mean(np.array(epoch_losses)))
        test_loss = 0
        accuracy = 0
        model.eval()

        ## get AUPRC scores
        with torch.no_grad():
            make_prediction_csv(model, PREDICTION_PATH, mode=TRAIN_MODE, embed=True,
                                test_path=test_dir)
            df_dict = evaluate(PREDICTION_PATH,
                               ANNOTATIONS_PATH,
                               YAML_PATH,
                               'coarse')
            df_dict_fine = evaluate(PREDICTION_PATH,
                               ANNOTATIONS_PATH,
                               YAML_PATH,
                               'fine')
            micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
            micro_auprc_fine, eval_df_fine = micro_averaged_auprc(df_dict_fine, return_df=True)
            print('Micro_AUPRC Coarse:', micro_auprc)
            print('Micro_AUPRC Fine:', micro_auprc_fine)
            if micro_auprc > best_micro_auprc_coarse or micro_auprc_fine > best_micro_auprc_fine:
                name, ext = os.path.splitext(PREDICTION_PATH)
                shutil.copy(PREDICTION_PATH, f'{name}_best_coarse={micro_auprc:.3f}_fine={micro_auprc_fine:.3f}{ext}')
                torch.save(model.state_dict(), f'models/{RUN_NAME}_coarse={micro_auprc:.3f}_fine={micro_auprc_fine:.3f}.ckpt')
                best_micro_auprc_coarse = micro_auprc
                best_micro_auprc_fine = micro_auprc_fine
                stagnation = 0
                print('Best so far')

            else:
                stagnation += 1
                print('Stagnation:', stagnation)

            print()
            model.train()