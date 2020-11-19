

import os
import sys, gc
import time
import copy
import random
import numpy as np
import matplotlib.pyplot as plt



import tensorflow as tf





# ==========================================================================================================
# Get CAP (Class Activation Map)
def get_class_map(label, net_cam, sig_size, gap_w):
    output_channels = int(net_cam.get_shape()[-1])   # channel 갯수
    w_transpose = tf.transpose(gap_w)
    w_label = tf.gather(w_transpose, label)
    w_label = tf.reshape(w_label, [-1, output_channels, 1])
    net_cam_image = tf.reshape(net_cam, [-1, net_cam.get_shape()[1], 1, output_channels])
    net_cam_resize = tf.image.resize_bilinear(net_cam_image, [sig_size, 1])
    net_cam_reshape = tf.reshape(net_cam_resize, [-1, sig_size * 1, output_channels])
    classmap = tf.matmul(net_cam_reshape, w_label)
    classmap = tf.reshape(classmap, [-1, sig_size])
    return classmap




# =====================================================================================================


class Model:
    def __init__(self, graph, model_name, device='',
                 sig_size=44100, n_classes=4, cl=3, dl=2, l_rate=1e-5,
                 ker=2, filter=8, node=16, pool=2, stride=2, cost_wgt=[], max_only=False):
        self.Graph = graph
        self.Device = device
        self.ModelName = model_name
        self.SIG_SIZE = sig_size
        print(model_name)
        #
        #-----------------------------------------
        # Make Network
        self.C_LAYERS = cl
        self.D_LAYERS = dl
        #
        KERNEL = ker
        FILTER = filter
        DENSE_NODE = node
        POOL = pool
        STRIDE = stride
        DROP_OUT = 0.7
        tf.set_random_seed(777)
        #
        self.X = tf.placeholder(tf.float32, [None, self.SIG_SIZE, 1])
        self.Y = tf.placeholder(tf.int64, [None, n_classes])
        self.IS_TRAIN = tf.placeholder(tf.bool)
        #
        net_sig = self.X
        ffwd = net_sig
        for n in range(self.C_LAYERS):
            print('net_sig', net_sig.shape)
            FILTER = FILTER * (n % 2 + 1)
            net_sig = tf.layers.conv1d(inputs=net_sig, filters=FILTER, kernel_size=KERNEL, padding='same', activation=tf.nn.relu)
            if not max_only:
                net_min = tf.layers.max_pooling1d(inputs=-net_sig, pool_size=POOL, strides=STRIDE, padding='same')
                net_avg = tf.layers.average_pooling1d(inputs=net_sig, pool_size=POOL, strides=STRIDE, padding='same')
            #
            net_sig = tf.layers.max_pooling1d(inputs=net_sig, pool_size=POOL, strides=STRIDE, padding='same')
            net_sig = tf.layers.batch_normalization(inputs=net_sig, center=True, scale=True, training=self.IS_TRAIN)
            net_sig = tf.layers.dropout(inputs=net_sig, rate=DROP_OUT, training=self.IS_TRAIN)
            #
            if not max_only:
                net_min = tf.layers.batch_normalization(inputs=net_min, center=True, scale=True, training=self.IS_TRAIN)
                net_min = tf.layers.dropout(inputs=net_min, rate=DROP_OUT, training=self.IS_TRAIN)
                #
                net_avg = tf.layers.batch_normalization(inputs=net_avg, center=True, scale=True, training=self.IS_TRAIN)
                net_avg = tf.layers.dropout(inputs=net_avg, rate=DROP_OUT, training=self.IS_TRAIN)
            #
            ffwd = tf.layers.max_pooling1d(inputs=ffwd, pool_size=POOL, strides=STRIDE, padding='same')
            #
            if not max_only:
                net_sig = tf.concat([net_sig, net_min, net_avg, ffwd], axis=2)
            else:
                net_sig = tf.concat([net_sig, ffwd], axis=2)
        #
        # -----------------------------------------------------------
        # Flatten and Concatenation
        print('net_sig:', net_sig)
        self.net_cam = net_sig
        self.net_flat = tf.reshape(net_sig, [-1, net_sig.shape[1]._value * net_sig.shape[2]._value])
        self.net_merge = self.net_flat
        for i in range(self.D_LAYERS):
            self.net_merge = tf.layers.dense(self.net_merge, DENSE_NODE, activation=tf.nn.relu)
            self.net_merge = tf.layers.dropout(self.net_merge, DROP_OUT)
            DENSE_NODE = DENSE_NODE * 2
        #
        self.logits = tf.layers.dense(self.net_merge, n_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.nn.softmax(self.logits)
        if cost_wgt != [] and len(cost_wgt) == self.logits.shape[1]._value:
            print('apply weight:', cost_wgt)
            self.logits = tf.multiply(self.logits, cost_wgt)
            self.logits = tf.nn.softmax(self.logits)
        #
        print('net_merge:', self.net_merge, ', net_cam:', self.net_cam)
        #
        # -----------------------------------------------------------
        # for CAM
        NET_DEPTH = self.net_cam.shape[2]._value
        self.gap = tf.reduce_mean(self.net_cam, (1))
        self.gap_w = tf.get_variable('cam_w1', shape=[NET_DEPTH, n_classes], initializer=tf.contrib.layers.xavier_initializer())
        self.cam = get_class_map(0, self.net_cam, self.SIG_SIZE, self.gap_w)
        #
        # -----------------------------------------------------------
        # Batch Normalization
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost)
        #
        # -----------------------------------------------------------
        # Prediction and Accuracy
        self.predict = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #
        # Make Session
        if self.Device != '':
            self.config = tf.ConfigProto()
            self.config.gpu_options.visible_device_list = self.Device
            self.sess = tf.Session(graph=self.Graph, config=self.config)
        else:
            self.sess = tf.Session(graph=self.Graph)
        #
        self.sess.run(tf.global_variables_initializer())
        if (model_name != ''):
            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_name)
            except Exception as ex:
                print(str(ex), model_name)
    # ------------------------------------------------------------------------------------------
    def train(self, x, y, is_train):
        if is_train:
            c, _ = self.sess.run([self.cost, self.optimizer],
                                 feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
            l, p, a = self.sess.run([self.logits, self.predict, self.accuracy],
                                    feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        else:
            c, l, p, a = self.sess.run([self.cost, self.logits, self.predict, self.accuracy],
                                    feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        return c, l, p, a
    #
    def test(self, x):
        l, p = self.sess.run([self.logits, self.predict], feed_dict={self.X: x, self.IS_TRAIN: False})
        return l, p
    #
    def get_cam(self, x):
        cam_val, p_val = self.sess.run([self.cam, self.predict], feed_dict={self.X: x, self.IS_TRAIN: False})
        return cam_val, p_val
    #
    def save(self, model_name):
        self.saver.save(self.sess, model_name)
        return


class Model_Features:
    def __init__(self, graph, model_name, device='',
                 n_features=4, n_classes=4, dl=2, node=16, l_rate=1e-5, cost_wgt=[]):
        self.Graph = graph
        self.Device = device
        self.ModelName = model_name
        print(model_name)
        #
        #-----------------------------------------
        # Make Network
        self.D_LAYERS = dl
        DENSE_NODE = node
        DROP_OUT = 0.7
        tf.set_random_seed(777)
        #
        self.X = tf.placeholder(tf.float32, [None, n_features])
        self.Y = tf.placeholder(tf.int64, [None, n_classes])
        self.IS_TRAIN = tf.placeholder(tf.bool)
        #
        self.net = self.X
        for i in range(self.D_LAYERS):
            self.net = tf.layers.dense(self.net, DENSE_NODE, activation=tf.nn.relu)
            self.net = tf.layers.batch_normalization(inputs=self.net, center=True, scale=True, training=self.IS_TRAIN)
            self.net = tf.layers.dropout(self.net, DROP_OUT)
            DENSE_NODE = DENSE_NODE * 2
        #
        self.logits = tf.layers.dense(self.net, n_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.nn.softmax(self.logits)
        if cost_wgt != [] and len(cost_wgt) == self.logits.shape[1]._value:
            print('apply weight:', cost_wgt)
            self.logits = tf.multiply(self.logits, cost_wgt)
            self.logits = tf.nn.softmax(self.logits)
        print('net:', self.net)
        #
        # -----------------------------------------------------------
        # Batch Normalization
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost)
        #
        # -----------------------------------------------------------
        # Prediction and Accuracy
        self.predict = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #
        # Make Session
        if self.Device != '':
            self.config = tf.ConfigProto()
            self.config.gpu_options.visible_device_list = self.Device
            self.sess = tf.Session(graph=self.Graph, config=self.config)
        else:
            self.sess = tf.Session(graph=self.Graph)
        #
        self.sess.run(tf.global_variables_initializer())
        if model_name != '':
            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_name)
            except Exception as ex:
                print(str(ex), model_name)
    # ------------------------------------------------------------------------------------------
    def train(self, x, y, is_train):
        if is_train:
            c, _ = self.sess.run([self.cost, self.optimizer],
                                 feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
            l, p, a = self.sess.run([self.logits, self.predict, self.accuracy],
                                    feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        else:
            c, l, p, a = self.sess.run([self.cost, self.logits, self.predict, self.accuracy],
                                    feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        return c, l, p, a
    #
    def test(self, x):
        l, p = self.sess.run([self.logits, self.predict], feed_dict={self.X: x, self.IS_TRAIN: False})
        return l, p
    #
    def save(self, model_name):
        self.saver.save(self.sess, model_name)
        #
        myFile = open('graph_def.txt', 'w')
        myFile.write(str(self.sess.graph_def))
        myFile.close()
        #
        tf.io.write_graph(self.sess.graph_def, '.', model_name.replace('.ckpt', '.pb'), as_text=False)
        tf.io.write_graph(self.sess.graph_def, '.', model_name.replace('.ckpt', '.pbtxt'))
        return



class Model_FFT:
    def __init__(self, graph, model_name, device='',
                 sig_size=44100, fft_size=512, n_classes=4, cl=3, dl=2, l_rate=1e-5,
                 ker=2, filter=8, node=16, pool=2, stride=2, cost_wgt=[]):
        self.Graph = graph
        self.Device = device
        self.ModelName = model_name
        self.SIG_SIZE = sig_size
        self.FFT_SIZE = fft_size
        print(model_name)
        #
        #-----------------------------------------
        # Make Network
        self.C_LAYERS = cl
        self.D_LAYERS = dl
        #
        KERNEL = ker
        FILTER = filter
        DENSE_NODE = node
        POOL = pool
        STRIDE = stride
        DROP_OUT = 0.7
        tf.set_random_seed(777)
        #
        self.X = tf.placeholder(tf.float32, [None, self.SIG_SIZE, 1])
        self.F = tf.placeholder(tf.float32, [None, self.FFT_SIZE, 1])
        self.Y = tf.placeholder(tf.int64, [None, n_classes])
        self.IS_TRAIN = tf.placeholder(tf.bool)
        #
        net_sig = self.X
        ffwd = net_sig
        net_fft = self.F
        ffwd_fft = net_fft
        for n in range(self.C_LAYERS):
            print('net_sig', net_sig.shape)
            net_sig = tf.layers.conv1d(inputs=net_sig, filters=FILTER, kernel_size=KERNEL, padding='same', activation=tf.nn.relu)
            net_min = tf.layers.max_pooling1d(inputs=-net_sig, pool_size=POOL, strides=STRIDE, padding='same')
            net_avg = tf.layers.average_pooling1d(inputs=net_sig, pool_size=POOL, strides=STRIDE, padding='same')
            #
            net_sig = tf.layers.max_pooling1d(inputs=net_sig, pool_size=POOL, strides=STRIDE, padding='same')
            net_sig = tf.layers.batch_normalization(inputs=net_sig, center=True, scale=True, training=self.IS_TRAIN)
            net_sig = tf.layers.dropout(inputs=net_sig, rate=DROP_OUT, training=self.IS_TRAIN)
            #
            net_min = tf.layers.batch_normalization(inputs=net_min, center=True, scale=True, training=self.IS_TRAIN)
            net_min = tf.layers.dropout(inputs=net_min, rate=DROP_OUT, training=self.IS_TRAIN)
            #
            net_avg = tf.layers.batch_normalization(inputs=net_avg, center=True, scale=True, training=self.IS_TRAIN)
            net_avg = tf.layers.dropout(inputs=net_avg, rate=DROP_OUT, training=self.IS_TRAIN)
            #
            ffwd = tf.layers.max_pooling1d(inputs=ffwd, pool_size=POOL, strides=STRIDE, padding='same')
            #
            net_sig = tf.concat([net_sig, net_min, net_avg, ffwd], axis=2)
        #
        for n in range(0, self.C_LAYERS, 2):
            print('net_fft', net_fft.shape)
            net_fft = tf.layers.conv1d(inputs=net_fft, filters=FILTER, kernel_size=KERNEL, padding='same', activation=tf.nn.relu)
            net_fft = tf.layers.max_pooling1d(inputs=net_fft, pool_size=POOL, strides=STRIDE, padding='same')
            net_fft = tf.layers.batch_normalization(inputs=net_fft, center=True, scale=True, training=self.IS_TRAIN)
            net_fft = tf.layers.dropout(inputs=net_fft, rate=DROP_OUT, training=self.IS_TRAIN)
            ffwd_fft = tf.layers.max_pooling1d(inputs=ffwd_fft, pool_size=POOL, strides=STRIDE, padding='same')
            net_fft = tf.concat([net_fft, ffwd_fft], axis=2)
        #
        # -----------------------------------------------------------
        # Flatten and Concatenation
        print('net_sig:', net_sig)
        self.net_cam = net_sig
        self.net_flat = tf.reshape(net_sig, [-1, net_sig.shape[1]._value * net_sig.shape[2]._value])
        self.net_flat_fft = tf.reshape(net_fft, [-1, net_fft.shape[1]._value * net_fft.shape[2]._value])
        #
        self.net_merge = tf.concat([self.net_flat, self.net_flat_fft], axis= 1)
        for i in range(self.D_LAYERS):
            self.net_merge = tf.layers.dense(self.net_merge, DENSE_NODE, activation=tf.nn.relu)
            self.net_merge = tf.layers.dropout(self.net_merge, DROP_OUT)
            DENSE_NODE = DENSE_NODE * 2
        #
        self.logits = tf.layers.dense(self.net_merge, n_classes, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.nn.softmax(self.logits)
        if cost_wgt != [] and len(cost_wgt) == self.logits.shape[1]._value:
            print('apply weight:', cost_wgt)
            self.logits = tf.multiply(self.logits, cost_wgt)
            self.logits = tf.nn.softmax(self.logits)
        print('net_merge:', self.net_merge, ', net_cam:', self.net_cam)
        #
        # -----------------------------------------------------------
        # for CAM
        NET_DEPTH = self.net_cam.shape[2]._value
        self.gap = tf.reduce_mean(self.net_cam, (1))
        self.gap_w = tf.get_variable('cam_w1', shape=[NET_DEPTH, n_classes], initializer=tf.contrib.layers.xavier_initializer())
        self.cam = get_class_map(0, self.net_cam, self.SIG_SIZE, self.gap_w)
        #
        # -----------------------------------------------------------
        # Batch Normalization
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost)
        #
        # -----------------------------------------------------------
        # Prediction and Accuracy
        self.predict = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #
        # Make Session
        if self.Device != '':
            self.config = tf.ConfigProto()
            self.config.gpu_options.visible_device_list = self.Device
            self.sess = tf.Session(graph=self.Graph, config=self.config)
        else:
            self.sess = tf.Session(graph=self.Graph)
        #
        self.sess.run(tf.global_variables_initializer())
        if (model_name != ''):
            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_name)
            except Exception as ex:
                print(str(ex), model_name)
    # ------------------------------------------------------------------------------------------
    def train(self, x, f, y, is_train):
        if is_train:
            c, _ = self.sess.run([self.cost, self.optimizer],
                                 feed_dict={self.X: x, self.F: f, self.Y: y, self.IS_TRAIN: is_train})
            l, p, a = self.sess.run([self.logits, self.predict, self.accuracy],
                                    feed_dict={self.X: x, self.F: f, self.Y: y, self.IS_TRAIN: is_train})
        else:
            c, l, p, a = self.sess.run([self.cost, self.logits, self.predict, self.accuracy],
                                    feed_dict={self.X: x, self.F: f, self.Y: y, self.IS_TRAIN: is_train})
        return c, l, p, a
    #
    def test(self, x, f):
        l, p = self.sess.run([self.logits, self.predict], feed_dict={self.X: x, self.F: f, self.IS_TRAIN: False})
        return l, p
    #
    def get_cam(self, x, f):
        cam_val, p_val = self.sess.run([self.cam, self.predict], feed_dict={self.X: x, self.F: f, self.IS_TRAIN: False})
        return cam_val, p_val
    #
    def save(self, model_name):
        self.saver.save(self.sess, model_name)
        return

