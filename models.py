# -*- coding: utf-8 -*-
import tensorflow as tf

# For input size input_nc x 256 x 256
class EncoderDecoder(object):
    def __init__(self, input_nc, output_nc, ngf):
        super(EncoderDecoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

    def __call__(self, input):
        #input_nc=3, output_nc=3
        #ngf=64
        #conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
        s = self.output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        with tf.variable_scope("generator") as scope:
            # image is (256 x 256 x 3)
            e1 = conv2d(image, self.ngf, name='g_e1_conv')
            # e1 is (128 x 128 x self.ngf)
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.ngf*2, name='g_e2_conv'))
            # e2 is (64 x 64 x self.ngf*2)
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.ngf*4, name='g_e3_conv'))
            # e3 is (32 x 32 x self.ngf*4)
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.ngf*8, name='g_e4_conv'))
            # e4 is (16 x 16 x self.ngf*8)
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.ngf*8, name='g_e5_conv'))
            # e5 is (8 x 8 x self.ngf*8)
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.ngf*8, name='g_e6_conv'))
            # e6 is (4 x 4 x self.ngf*8)
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.ngf*8, name='g_e7_conv'))
            # e7 is (2 x 2 x self.ngf*8)
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.ngf*8, name='g_e8_conv'))
            # e8 is (1 x 1 x self.ngf*8)

            # d1 is (2 x 2 x self.ngf*8*2)
            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.ngf*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)
            # d2 is (4 x 4 x self.ngf*8*2)
            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.ngf*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)
            # d3 is (8 x 8 x self.ngf*8*2)
            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.ngf*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)
            # d4 is (16 x 16 x self.ngf*8*2)
            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.ngf*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)
            # d5 is (32 x 32 x self.ngf*4*2)
            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.ngf*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)
            # d6 is (64 x 64 x self.ngf*2*2)
            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.ngf*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)
            # d7 is (128 x 128 x self.ngf*1*2)
            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.ngf], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)
            # d8 is (256 x 256 x output_nc)
            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_nc], name='g_d8', with_w=True)
        return tf.nn.tanh(self.d8)

