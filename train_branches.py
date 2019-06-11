from datetime import datetime

from poutyne.framework import Model
from poutyne.framework.callbacks import *

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from load_dataset import AudioDatasetFine, label_hierarchy

DATE = datetime.now().strftime('%Y%m%d_%H%M%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dir = r"D:\DCASE_2019\audio\augmented\spec_vgg\train"
test_dir = r"D:\DCASE_2019\audio\augmented\spec_vgg\validate"

coarse_index = 0

BATCH_SIZE = 32
label_start, label_end = label_hierarchy[coarse_index + 1]
NUM_CLASSES = len(range(label_start, label_end))
print('number of classes:', NUM_CLASSES)

TRAIN = AudioDatasetFine(train_dir, coarse_index)
TEST = AudioDatasetFine(test_dir, coarse_index)

TRAIN_LOADER = DataLoader(dataset=TRAIN, batch_size=BATCH_SIZE, shuffle=True)
TEST_LOADER = DataLoader(dataset=TEST, batch_size=BATCH_SIZE, shuffle=True)


# train_sampler = torch.utils.data.sampler.WeightedRandomSampler(TRAIN_WEIGHTS, 2351)
# test_sampler = torch.utils.data.sampler.WeightedRandomSampler(TEST_WEIGHTS, 443)


class ConvBlock(nn.Module):
    """This creates a convolutional layer with optional maxpool, batchnorm, and dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                 batchnorm=True, maxpool=True, maxpool_size=(2, 2), dropout=None):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding) # , bias=False ?
        # print('kernel', kernel_size, stride, padding, maxpool)
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


class VGG_alt(nn.Module):
    """Based on AudioSet paper, with some maxpool size modifications"""
    def __init__(self):
        super(VGG_alt, self).__init__()

        DROPOUT = .5
        self.emb_size = 49152

        # spectrogram convolutions
        self.conv_block_1 = ConvBlock(in_channels=1, out_channels=8, kernel_size=(1, 1), stride=(1, 1),
                                       padding=(0, 0), batchnorm=True, maxpool=False, maxpool_size=(2, 16), dropout=DROPOUT)

        self.conv_block_2 = ConvBlock(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=(2, 2), dropout=DROPOUT)

        self.conv_block_3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=(16, 128), stride=(4, 16),
                                       padding=(8, 16), batchnorm=True, maxpool=True, maxpool_size=(4, 4), dropout=DROPOUT)

        self.conv_block_4 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=(2, 2), dropout=DROPOUT)

        self.conv_block_5 = ConvBlock(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=None, dropout=DROPOUT)

        self.conv_block_6 = ConvBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                                       padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=(2, 4), dropout=DROPOUT)

        # self.conv_block_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
        #                               padding=(1, 1), batchnorm=True, maxpool=False, maxpool_size=(2, 4), dropout=DROPOUT)


        # openl3 embedding convolutions
        # self.emb_conv_1 = ConvBlock(in_channels=1, out_channels=4, kernel_size=(5, 5), stride=(2, 2),
        #                             padding=(1, 1), batchnorm=True, maxpool=True, maxpool_size=(4, 4), dropout=DROPOUT)
        # self.emb_conv_2 = ConvBlock(in_channels=4, out_channels=8, kernel_size=(5, 5), stride=(2, 2),
        #                             padding=(1, 1), batchnorm=True, maxpool=True, maxpool_size=(2, 2),
        #                             dropout=DROPOUT)
        # self.emb_conv_3 = ConvBlock(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=(2, 2),
        #                             padding=(1, 1), batchnorm=True, maxpool=True, maxpool_size=(2, 2),
        #                             dropout=DROPOUT)
        # self.emb_conv_4 = ConvBlock(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(2, 2),
        #                             padding=(1, 1), batchnorm=True, maxpool=True, maxpool_size=(2, 2),
        #                             dropout=DROPOUT)

        # fc layers
        # self.fc_emb1 = nn.Linear(self.emb_size, 2**10, bias=True)
        # self.fc_emb2 = nn.Linear(2**10, 2**8, bias=True)
        self.fc1 = nn.Bilinear(256, 1280, 512, bias=True)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc2_bn = nn.BatchNorm1d(256)
        # self.fc3 = nn.Linear(2**7, 2**6, bias=True)
        # self.fc4 = nn.Linear(2**8, 2**6, bias=True)
        self.fc_final = nn.Linear(256, NUM_CLASSES, bias=True)

        self.dropout = nn.Dropout(.2)

        # self.init_weights()

    # def init_weights(self):
    #     init_layer(self.fc)

    def forward(self, nn_input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        # x, emb, vgg = nn_input
        x, vgg = nn_input
        '''(batch_size, 1, times_steps, freq_bins)'''

        # spectrogram convolutions
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        # x = self.conv_block_7(x)

        # openl3 convolutions
        # emb = self.emb_conv_1(emb)
        # emb = self.emb_conv_2(emb)
        # emb = self.emb_conv_3(emb)
        # emb = self.emb_conv_4(emb)

        # reshape for fc layers
        x = x.view(x.size(0), -1)
        # emb = emb.view(emb.size(0), -1)
        vgg = vgg.view(vgg.size(0), -1)
        # print(x.shape, emb.shape)
        # print(x.shape)
        # emb = self.fc_emb1(emb)
        # emb = self.fc_emb2(emb)

        # takes spectrogram and openl3 conv outputs
        x = self.fc1(x, vgg)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        # x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc4(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc5(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc6(x))
        # x = self.dropout(x)
        x = self.fc_final(x)
        # output = torch.sigmoid(x)
        output = x
        return output


if __name__ == '__main__':

    # model = NeuralNetwork().to(device)
    # model = VGG_11().to(device)
    model = VGG_alt().to(device)
    # model = OpenL3().to(device)

    ## if training from checkpoint; ensure checkpoint matches model class architecture
    # checkpoint = torch.load("models/20190531_151918_best_epoch_19_val_loss=0.1182.ckpt")
    # model.load_state_dict(checkpoint)

    # Loss and optimizer
    # criterion = nn.BCELoss()  # must be this for multi-label predictions
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array(TRAIN_WEIGHTS).astype(np.float32)**.2).to(device))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    # to Poutyne
    model = Model(model, optimizer, criterion, metrics=['bin_acc'])

    # Callbacks
    tb_writer = SummaryWriter('./tensorboard/{}'.format(DATE))

    callbacks = [
        # Save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint('./models/%s_last_epoch.ckpt' % DATE, temporary_filename='last_epoch.ckpt.tmp'),

        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint('./models/%s_best_epoch_{epoch}_val_loss={val_loss:.4f}.ckpt' % DATE, monitor='val_loss', mode='min',
                        save_best_only=True, restore_best=True,
                        verbose=True, temporary_filename='best_epoch.ckpt.tmp'),

        # Save the losses and accuracies for each epoch in a TSV.
        CSVLogger('./models/log.tsv', separator='\t'),

        ReduceLROnPlateau(patience=5, verbose=True, factor=0.1),
        EarlyStopping(patience=10, verbose=True),
        TerminateOnNaN(),
        # policies.sgdr_phases(6, 6, lr=(1.0, 0.1), cycle_mult = 2) # doesn't work as callback
    ]

    save_file_path = './models/weights.{epoch:02d}-{val_loss:.4f}.txt'
    save_best_model = PeriodicSaveCallback(save_file_path, temporary_filename='./tmp/file.txt', atomic_write=False,
                                           save_best_only=True, verbose=True)

    # Train the model
    model.fit_generator(TRAIN_LOADER, TEST_LOADER, epochs=10000, callbacks=callbacks)