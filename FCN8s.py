from Utils import *


class FCN8s(tf.keras.layers.Layer):
    def __init__(self, class_num):
        super(FCN8s, self).__init__()
        self.layer1 = ConvBlock(filter_num=64)
        self.layer2 = ConvBlock(filter_num=64)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer3 = ConvBlock(filter_num=128)
        self.layer4 = ConvBlock(filter_num=128)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer5 = ConvBlock(filter_num=256)
        self.layer6 = ConvBlock(filter_num=256)
        self.layer7 = ConvBlock(filter_num=256)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer8 = ConvBlock(filter_num=512)
        self.layer9 = ConvBlock(filter_num=512)
        self.layer10 = ConvBlock(filter_num=512)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer11 = ConvBlock(filter_num=512)
        self.layer12 = ConvBlock(filter_num=512)
        self.layer13 = ConvBlock(filter_num=512)
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.fcn1 = tf.keras.layers.Conv2D(4096, (7, 7), padding='same', kernel_initializer='he_normal')
        self.relu1 = tf.keras.layers.Activation(activation='relu')
        self.fcn2 = tf.keras.layers.Conv2D(4096, (1, 1), padding='same', kernel_initializer='he_normal')
        self.relu2 = tf.keras.layers.Activation(activation='relu')
        self.fcn3 = tf.keras.layers.Conv2D(class_num, (1, 1), padding='same', kernel_initializer='he_normal')
        self.relu3 = tf.keras.layers.Activation(activation='relu')
        self.conv1 = tf.keras.layers.Conv2D(class_num, (1, 1), padding='same', kernel_initializer='he_normal')
        self.conv2 = tf.keras.layers.Conv2D(class_num, (1, 1), padding='same', kernel_initializer='he_normal')
        self.deconv1 = tf.keras.layers.Conv2DTranspose(class_num, (2, 2), strides=(2, 2), padding='valid', output_padding=(1, 1), kernel_initializer='he_normal')
        self.deconv2 = tf.keras.layers.Conv2DTranspose(class_num, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')
        self.deconv3 = tf.keras.layers.Conv2DTranspose(class_num, (16, 16), strides=(8, 8), padding='same', kernel_initializer='he_normal')
        self.softmax = tf.keras.layers.Activation(activation='softmax')

    def call(self, inputs):
        # encoder
        # 144x144
        x1 = self.layer1(inputs)
        x1 = self.layer2(x1)
        # 72x72
        x1 = self.pool1(x1)
        x2 = self.layer3(x1)
        x2 = self.layer4(x2)
        # 36x36
        x2 = self.pool2(x2)
        x3 = self.layer5(x2)
        x3 = self.layer6(x3)
        x3 = self.layer7(x3)
        # 18x18
        x3 = self.pool3(x3)
        x3_score = self.conv1(x3)
        x4 = self.layer8(x3)
        x4 = self.layer9(x4)
        x4 = self.layer10(x4)
        # 9x9
        x4 = self.pool4(x4)
        x4_score = self.conv2(x4)
        x5 = self.layer11(x4)
        x5 = self.layer12(x5)
        x5 = self.layer13(x5)
        # 4x4
        x5 = self.pool5(x5)
        x6 = self.fcn1(x5)
        x6 = self.relu1(x6)
        x7 = self.fcn2(x6)
        x7 = self.relu2(x7)
        x8 = self.fcn3(x7)
        x8 = self.relu3(x8)

        # decoder
        # 9x9
        x9 = self.deconv1(x8)
        x9 = tf.keras.layers.add(inputs=[x9, x4_score])
        # 18x18
        x10 = self.deconv2(x9)
        x10 = tf.keras.layers.add(inputs=[x10, x3_score])
        # 144x144
        x11 = self.deconv3(x10)
        output = self.softmax(x11)
        # print(output.shape)

        return output