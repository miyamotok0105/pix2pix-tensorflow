# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import tensorflow as tf
# from models import EncoderDecoder, Discriminator
from data import get_training_set, get_test_set

parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
parser.add_argument('--dataset', '-i', default='facades', help='Directory of image files.')
args = parser.parse_args()

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + args.dataset)
test_set = get_test_set(root_path + args.dataset)

for train in train_set:
	print(train[0].shape)
	print(train[1].shape)
	break

#https://github.com/yenchenlin/pix2pix-tensorflow/blob/master/model.py
# real_data = tf.placeholder(tf.float32,
#                             [self.batch_size, self.image_size, self.image_size,
#                              self.input_c_dim + self.output_c_dim],
#                             name='real_A_and_B_images')

# real_B = real_data[:, :, :, :self.input_c_dim]
# real_A = real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

# self.fake_B = self.generator(self.real_A)

# self.real_AB = tf.concat([self.real_A, self.real_B], 3)
# self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
# self.D, self.D_logits = self.discriminator(self.real_AB, reuse=False)
# self.D_, self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

# self.fake_B_sample = self.sampler(self.real_A)

# self.d_sum = tf.summary.histogram("d", self.D)
# self.d__sum = tf.summary.histogram("d_", self.D_)
# self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

# self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
# self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
# self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
#                 + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

# self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
# self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

# self.d_loss = self.d_loss_real + self.d_loss_fake

# self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
# self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

# t_vars = tf.trainable_variables()

# self.d_vars = [var for var in t_vars if 'd_' in var.name]
# self.g_vars = [var for var in t_vars if 'g_' in var.name]

# self.saver = tf.train.Saver()
