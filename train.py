

import os
import random
import numpy as np
import pandas as pd


import librosa


import tensorflow as tf
import models




#===============================================================================================
# Variables
DATA_PATH = 'dataset/'   # only Full-Path available
MODEL_PATH = 'models/'
RESULT_PATH = 'results/'
TRAIN_PATH = 'dataset/train/'
VALID_PATH = 'dataset/valid/'



NUM_CLASSES = 4
SAMPLE_RATE = 22050
UNIT_SEC = 2
SIG_SIZE = SAMPLE_RATE * UNIT_SEC  # 2sec



C_LAYERS = 8
D_LAYERS = 2
KERNEL = 16
FILTER = 8
D_NODE = 8
POOL = 2
STRIDE = 2


Device = ''
START = 0
LOAD_SIZE = 1000






# ===============================================================================================================
# Training
def get_audio_data(DATA_PATH, audio_size=44100, data_size=100, sr=44100):
    labels = os.listdir(DATA_PATH)
    labels.sort()
    #
    batch_audio = []
    batch_fft = []
    batch_y = []
    #
    for dirname, subdirs, files in os.walk(DATA_PATH):
        class_name = dirname.split('/')[-1]
        try:
            y = labels.index(class_name)
        except Exception as ex:
            #print(str(ex))
            continue
        # ------------------------------------------------------------------------
        # file path와 label_index 저장
        temp_list = []
        for audio_file in files:
            temp_list.append([dirname + '/' + audio_file, y])
        if len(temp_list) == 0:
            continue
        #
        # ------------------------------------------------------------------------
        for i in range(data_size):
            index = np.random.randint(0, len(temp_list))
            audio_file, y = temp_list[index]
            #
            sigs, s_rate = librosa.load(audio_file, sr=sr)
            sigs_norm = sigs[: audio_size] / np.max(sigs[: audio_size])
            #
            batch_audio.append(sigs_norm)
            label = [0, 0, 0, 0]
            label[y] = 1    # make one-hot label
            batch_y.append(label)
    #
    batch_audio = np.array(batch_audio)
    batch_y = np.array(batch_y)
    return batch_audio, batch_y



def train_model(model, cost, acc, dataset, batch_size=1000):
    best_cost = cost
    best_acc = acc
    print('' * 25)
    #
    # Each epoch has a training and validation phase
    train_acc = 0
    for phase in ['train', 'val']:
        if phase == 'train':
            IS_TRAIN = True
        else:
            IS_TRAIN = False
        # ------------------------------------------------------------------
        batch_audio, batch_labels = dataset.get(phase)
        if batch_audio is None:
            continue
        #
        # Training / Evaluation
        cost_sum = 0
        accuracy_sum = 0
        total_cnt = 0
        #
        BATCH_CNT = int(len(batch_audio) / batch_size)
        for b in range(BATCH_CNT):
            batch_X = batch_audio[batch_size * b: batch_size * (b + 1)].reshape(-1, SIG_SIZE, 1)
            batch_Y = batch_labels[batch_size * b: batch_size * (b + 1)]
            #print(len(batch_X))
            if len(batch_X) == 0:
                continue
            #
            c, l, p, a = model.train(batch_X, batch_Y, IS_TRAIN)
            cost_sum += c
            accuracy_sum += a
            total_cnt += 1
        #
        if total_cnt == 0:
            continue
        # ------------------------------------------------------
        print(total_cnt)
        epoch_cost = cost_sum / total_cnt
        epoch_accuracy = accuracy_sum / total_cnt
        #
        print(
            phase, ', cost={0:0.3f}'.format(epoch_cost),
            ', accuracy={0:0.3f}'.format(epoch_accuracy)
        )
        #
        # -------------------------------------------------------
        # Save Model
        if phase == 'val' and epoch_cost <= best_cost:
            print(model.ModelName)
            print('change best_model: {0:0.3f}'.format(epoch_accuracy))
            best_cost = epoch_cost
            best_acc = epoch_accuracy
            model.save(model.ModelName)
            #with open('config.json', 'w') as fp:
            #    json.dump(weight_keys, fp)
    return model, best_cost, best_acc




# ===============================================================================================================
# Create Model
PROJECT = 'voice_classifier'
learning_rate = 1e-6
cost_wgt = [1, 1, 1, 1]




g = tf.Graph()
with g.as_default():
    model = models.Model(g, MODEL_PATH + '{}.ckpt'.format(PROJECT), Device,
                         sig_size=SIG_SIZE, n_classes=NUM_CLASSES, cl=C_LAYERS, dl=D_LAYERS, l_rate=learning_rate,
                         ker=KERNEL, filter=FILTER, node=D_NODE, pool=POOL, stride=STRIDE,
                         cost_wgt=cost_wgt)




# --------------------------------------------------------------------------------------------------
best_cost = 1000
best_acc = 0.2
START_EPOCH = 0
NUM_EPOCHS = 10000
load_size = 100     # x4
batch_size = 100


for e in range(START_EPOCH, NUM_EPOCHS):
    print()
    print(e)
    # ----------------------------------------------
    # def get_audio_fft_data(DATA_PATH, audio_size=44100, fft_size=512, data_size=100, sr=44100, nfft=32768):
    #
    # Train data
    train_data = get_audio_data(TRAIN_PATH, data_size=load_size)
    #
    # Valid data
    valid_data = get_audio_data(VALID_PATH, data_size=load_size)
    #
    # --------------------------------------------------------------------
    my_dataset = {'train': train_data, 'val': valid_data}
    model, best_cost, best_acc = train_model(model, best_cost, best_acc, my_dataset, batch_size)
    print('{}, best_acc: {}'.format(e, best_acc))



