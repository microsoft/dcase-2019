import argparse

from datetime import datetime
import gc

import joblib

from poutyne.framework import Model
from poutyne.framework.callbacks import *

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# from load_dataset_20190710_5_two_parts_one_model import AudioDatasetFine, label_hierarchy
from load_dataset_20190711_1_one_part_one_model import AudioDatasetFine, label_hierarchy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#train_dir = r"D:\datasets\dcase5_processed\spec_vgg\train"
#test_dir = r"D:\datasets\dcase5_processed\spec_vgg\validate"
train_dir = r"/dcase/spec_vgg/train"
test_dir = r"/dcase/spec_vgg/validate"

MODEL_BASE = r'/dcase/output/models'
TENSORBOARD_BASE = r'/dcase/output/tensorboard'
# MODEL_BASE_Jianyu = r'/dcase/output/models_Jianyu/split-2-parts-train-1-model'
MODEL_BASE_Jianyu = r'/dcase/output/models_Jianyu/one-part-1-model-logmdel'

os.makedirs(MODEL_BASE, exist_ok=True)
os.makedirs(TENSORBOARD_BASE, exist_ok=True)
os.makedirs(MODEL_BASE_Jianyu, exist_ok=True)

index_to_files_dict_train = joblib.load('/dcase/spec_vgg/label_to_files_train.zip')
index_to_files_dict_test = joblib.load('/dcase/spec_vgg/label_to_files_test.zip')

NUM_COARSE_LABELS = 8
BATCH_SIZE = 64
MAX_EPOCHS = 100

USE_EXAMPLE_WEIGHTS = True

if USE_EXAMPLE_WEIGHTS:
    weights_fine = joblib.load('weights_fine_train.pkl')


class ConvBlock(nn.Module):
    """This creates a convolutional layer with optional maxpool, batchnorm, and dropout"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 batchnorm=True,
                 maxpool=True,
                 maxpool_size=(2, 2),
                 dropout=None):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)  # , bias=False ?
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

    def __init__(self, num_classes):
        super(VGG_alt, self).__init__()

        self.NUM_CLASSES = num_classes

        DROPOUT = .55
        self.emb_size = 49152

        self.conv_block_1 = ConvBlock(in_channels=1,
                                      out_channels=8,
                                      kernel_size=(1, 1),
                                      stride=(2, 2),
                                      padding=(0, 0),
                                      batchnorm=True,
                                      maxpool=False,
                                    #   maxpool_size=(2, 16),
                                      maxpool_size=(2, 8),
                                      dropout=DROPOUT)

        self.conv_block_2 = ConvBlock(in_channels=8,
                                      out_channels=16,
                                      kernel_size=(3, 3),
                                      stride=(2, 2),
                                      padding=(1, 1),
                                      batchnorm=True,
                                      maxpool=False,
                                      maxpool_size=(3, 3),
                                      dropout=DROPOUT)

        self.conv_block_3 = ConvBlock(in_channels=16,
                                      out_channels=32,
                                    #   kernel_size=(16, 128),
                                      kernel_size=(3, 3),
                                    #   stride=(4, 16),
                                    #   padding=(8, 16),
                                      stride=(4, 4),
                                      padding=(1, 1),
                                      batchnorm=True,
                                      maxpool=True,
                                    #   maxpool_size=(4, 4),
                                      maxpool_size=(4, 4),
                                      dropout=DROPOUT)

        self.conv_block_4 = ConvBlock(in_channels=32,
                                      out_channels=64,
                                      kernel_size=(3, 3),
                                      stride=(3, 3),
                                      padding=(1, 1),
                                      batchnorm=True,
                                      maxpool=False,
                                      maxpool_size=(3, 3),
                                      dropout=DROPOUT)

        self.conv_block_5 = ConvBlock(in_channels=64,
                                      out_channels=128,
                                      kernel_size=(3, 3),
                                      stride=(3, 3),
                                      padding=(1, 1),
                                      batchnorm=True,
                                      maxpool=False,
                                      maxpool_size=None,
                                      dropout=DROPOUT)

        self.conv_block_6 = ConvBlock(in_channels=128,
                                      out_channels=256,
                                      kernel_size=(3, 3),
                                      stride=(3, 3),
                                      padding=(1, 1),
                                      batchnorm=True,
                                      maxpool=False,
                                      maxpool_size=(4, 4),
                                      dropout=DROPOUT)


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
        # self.fc1 = nn.Bilinear(256, 1280, 512, bias=True)
        # self.fc1 = nn.Bilinear(256, 128, 512, bias=True)
        self.fc1 = nn.Bilinear(256, 1280, 512, bias=True)
        self.fc1_bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc2_bn = nn.BatchNorm1d(256)
        # self.fc3 = nn.Linear(2**7, 2**6, bias=True)
        # self.fc4 = nn.Linear(2**8, 2**6, bias=True)
        self.fc_final = nn.Linear(256, self.NUM_CLASSES, bias=True)

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
        # print('x.shape begining')
        # print('x.shape begining')
        # print('x.shape begining')
        # print(x.shape)
        # print('vgg.shape begining')
        # print('vgg.shape begining')
        # print('vgg.shape begining')
        # print(vgg.shape)

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
        # print('x.shape')
        # print('x.shape')
        # print('x.shape')
        # print(x.shape)
        # print('vgg.shape')
        # print('vgg.shape')
        # print('vgg.shape')
        # print(vgg.shape)
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


def get_label_range(coarse_index):
    label_start, label_end = label_hierarchy[coarse_index + 1]
    NUM_CLASSES = len(range(label_start, label_end))
    return label_start, label_end, NUM_CLASSES

  

global_index = 100

def build_Model(coarse_index, DATE):

    ## save the generated model to seperate corresponding to each window 
    # output_dir = os.path.join(MODEL_BASE_Jianyu, str(window_index))
    output_dir = MODEL_BASE_Jianyu
    print('output_dir output_dir')
    print(output_dir)

    abel_start, label_end, NUM_CLASSES = get_label_range(coarse_index)

    # model = NeuralNetwork().to(device)
    # model = VGG_11().to(device)
    model_tmp = VGG_alt(NUM_CLASSES).to(device)
    # model = OpenL3().to(device)

    ## if training from checkpoint; ensure checkpoint matches model class architecture
    # checkpoint = torch.load("models/20190531_151918_best_epoch_19_val_loss=0.1182.ckpt")
    # model.load_state_dict(checkpoint)

    # Loss and optimizer
    # criterion = nn.BCELoss()  # must be this for multi-label predictions
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array(TRAIN_WEIGHTS).astype(np.float32)**.2).to(device))

    if USE_EXAMPLE_WEIGHTS:
        weights = weights_fine[coarse_index]
        print(f'Using sample weights: {weights}')
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array(weights).astype(np.float32)).to(device))
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array(weights_fine[coarse_index]).astype(np.float32)**.2).to(device))
    else:
        print('Not using sample weights.')

    criterion = nn.BCEWithLogitsLoss()

    # optimizer = torch.optim.Adam(model_tmp.parameters(), lr=.001)
    optimizer = torch.optim.RMSprop(model_tmp.parameters())

    # to Poutyne
    model = Model(model_tmp, optimizer, criterion, metrics=['bin_acc'])

    # Callbacks
    tb_writer = SummaryWriter(os.path.join(TENSORBOARD_BASE, f'{DATE}_coarse={coarse_index}'))

    callbacks = [
        # Save the latest weights to be able to continue the optimization at the end for more epochs.
        ModelCheckpoint(
            os.path.join(output_dir, f'{DATE}_coarse={coarse_index}_last_epoch.ckpt'),

            temporary_filename=os.path.join(output_dir, 'last_epoch.ckpt.tmp')),

        # Save the weights in a new file when the current model is better than all previous models.
        ModelCheckpoint(
            # os.path.join(output_dir, '%s_coarse=%d_window=%d_best_epoch_{epoch}_val_loss={val_loss:.4f}.ckpt'% (DATE, coarse_index, global_index,)),
            os.path.join(output_dir, '%s_coarse=%d_best_epoch_{epoch}_val_loss={val_loss:.4f}.ckpt'% (DATE, coarse_index)),
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            restore_best=False,  #True
            verbose=True,
            temporary_filename=os.path.join(output_dir, 'best_epoch.ckpt.tmp')),

        # Save the losses and accuracies for each epoch in a TSV.
        CSVLogger(os.path.join(output_dir, f'{DATE}_coarse={coarse_index}_log.tsv'),
                  separator='\t'),
        # ReduceLROnPlateau(patience=5, verbose=True, factor=0.1),
        ReduceLROnPlateau(patience=6, verbose=True, factor=0.2),
        EarlyStopping(patience=10, verbose=True),
        TerminateOnNaN(),
        # policies.sgdr_phases(6, 6, lr=(1.0, 0.1), cycle_mult = 2) # doesn't work as callback
    ]

    save_file_path = os.path.join(output_dir, '%s_coarse=%d_weights.{epoch:02d}-{val_loss:.4f}.txt' % (DATE, coarse_index))
    save_best_model = PeriodicSaveCallback(save_file_path,
                                           temporary_filename=os.path.join(output_dir, 'tmp_file.txt'),
                                           atomic_write=False,
                                           save_best_only=True,
                                           verbose=True)


    return model, MAX_EPOCHS, callbacks
    # Train the model
    # model.fit_generator(TRAIN_LOADER, TEST_LOADER, epochs=MAX_EPOCHS, callbacks=callbacks)

    # del optimizer
    # del model
    # del model_tmp


def get_DataLoaders(coarse_index):
    label_start, label_end, NUM_CLASSES = get_label_range(coarse_index)
    print('label_start:', label_start)
    print('label_end:', label_end)
    print('number of classes:', NUM_CLASSES)

    if NUM_CLASSES < 2:
        print('Skipping this coarse category.')
        return

    # for i in range(0, len(index_to_files_dict_train)):
        # print(index_to_files_dict_train[i][0:9])
        # print(i)
        # print("len(index_to_files_dict_train[i])")
        # print(index_to_files_dict_train[i])
    TRAIN = AudioDatasetFine(train_dir, coarse_index, index_to_files_dict_train)
    TEST = AudioDatasetFine(test_dir, coarse_index, index_to_files_dict_test)

    # print(len(TRAIN))
    # print(len(TEST))

    # TRAIN_LOADER = DataLoader(dataset=TRAIN, batch_size=BATCH_SIZE, shuffle=True)
    # TEST_LOADER = DataLoader(dataset=TEST, batch_size=BATCH_SIZE, shuffle=True)

    # del TEST
    # del TRAIN
    # return TRAIN_LOADER, TEST_LOADER
    return TRAIN, TEST


# def build_Model(coarse_index, TRAIN_LOADER, TEST_LOADER, DATE):
# def build_Model(coarse_index, window_index, DATE):

def train_model(coarse_index, DATE):

    model, MAX_EPOCHS, callbacks = build_Model(coarse_index, DATE)

    # window_index = 0
    # TRAIN_LOADER, TEST_LOADER = get_DataLoaders(coarse_index, window_index)

    # for window_index in range(0, 2):
    #     # print(window_index)
    #     global_index = window_index
    #     # print('global_index ')
    #     print('global_index ')
    #     print('global_index ')
    #     print(global_index)
    #     # model, MAX_EPOCHS, callbacks = build_Model(coarse_index, window_index, DATE)
    #     TRAIN_LOADER, TEST_LOADER = get_DataLoaders(coarse_index, window_index)
    #     model.fit_generator(TRAIN_LOADER, TEST_LOADER, epochs=MAX_EPOCHS, callbacks=callbacks)
    #     del TEST_LOADER
    #     del TRAIN_LOADER
    #     print('finish one window data')
    # del TEST
    # del TRAIN

    TRAIN, TEST = get_DataLoaders(coarse_index)

    # print(len(TRAIN_0))
    # print(len(TEST_0))
    # print(len(TRAIN_1))
    # print(len(TEST_1))
    print('TRAIN')
    print(len(TRAIN))
    print('TEST')
    print(len(TEST))

    TRAIN_LOADER = DataLoader(dataset=TRAIN, batch_size=BATCH_SIZE, shuffle=True)
    TEST_LOADER = DataLoader(dataset=TEST, batch_size=BATCH_SIZE, shuffle=True)
    model.fit_generator(TRAIN_LOADER, TEST_LOADER, epochs=MAX_EPOCHS, callbacks=callbacks)
    del TEST
    del TRAIN


def print_gpu_ram():
    print(f'GPU memory allocated: {torch.cuda.memory_allocated()}')
    print(f'GPU memory cached: {torch.cuda.memory_cached()}')

    # for obj in gc.get_objects():
    #     try:
    #         if torch.is_tensor(obj) or (hasattr(obj, 'data')
    #                                     and torch.is_tensor(obj.data)):
    #             print(type(obj), obj.size())
    #             del obj
    #     except:
    #         pass


def main(coarse_category_idx, DATE):
    global BATCH_SIZE

    print(
        f'\n*****************\nTraining model for coarse category {coarse_category_idx}\n*******\n'
    )
    print_gpu_ram()

    # Hack to avoid batch-size 1 in final batch, which causes crash in batch-norm.
    # TODO: Should fix in Poutyne training loop to skip final batch when this happens.
    if coarse_category_idx == 3:
        BATCH_SIZE = 63

    print('Training')
    train_model(coarse_category_idx, DATE)

    print('Done training.')
    print_gpu_ram()

    print('Clearing GPU ram')
    torch.cuda.empty_cache()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Train model for a single coarse category.')

    parser.add_argument('index', type=int, help='coarse category index')
    parser.add_argument('date', type=str, help='date string')

    args = parser.parse_args()

    main(args.index, args.date)
