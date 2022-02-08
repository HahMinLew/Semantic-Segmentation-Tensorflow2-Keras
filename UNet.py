from Utils import *


class UNet(tf.keras.layers.Layer):
    def __init__(self, class_num):
        super(UNet, self).__init__()
        self.layer1 = ConvBlock(filter_num=64)
        self.layer2 = ConvBlock(filter_num=64)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer3 = ConvBlock(filter_num=128)
        self.layer4 = ConvBlock(filter_num=128)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer5 = ConvBlock(filter_num=256)
        self.layer6 = ConvBlock(filter_num=256)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer7 = ConvBlock(filter_num=512)
        self.layer8 = ConvBlock(filter_num=512)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer9 = ConvBlock(filter_num=1024)
        self.layer10 = ConvBlock(filter_num=1024)
        self.conv1 = tf.keras.layers.Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal')
        self.up1 = tf.keras.layers.UpSampling2D((2, 2))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation(activation='relu')
        self.layer11 = ConvBlock(filter_num=512)
        self.layer12 = ConvBlock(filter_num=512)
        self.conv2 = tf.keras.layers.Conv2D(256, (2, 2), padding='same', kernel_initializer='he_normal')
        self.up2 = tf.keras.layers.UpSampling2D((2, 2))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation(activation='relu')
        self.layer13 = ConvBlock(filter_num=256)
        self.layer14 = ConvBlock(filter_num=256)
        self.conv3 = tf.keras.layers.Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')
        self.up3 = tf.keras.layers.UpSampling2D((2, 2))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.Activation(activation='relu')
        self.layer15 = ConvBlock(filter_num=128)
        self.layer16 = ConvBlock(filter_num=128)
        self.conv4 = tf.keras.layers.Conv2D(64, (2, 2), padding='same', kernel_initializer='he_normal')
        self.up4 = tf.keras.layers.UpSampling2D((2, 2))
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.Activation(activation='relu')
        self.layer17 = ConvBlock(filter_num=64)
        self.layer18 = ConvBlock(filter_num=64)
        self.conv5 = tf.keras.layers.Conv2D(class_num, (1, 1), padding='same', kernel_initializer='he_normal')
        self.softmax = tf.keras.layers.Activation(activation='softmax')

    def call(self, inputs, training=False):
        # encoder
        # 144x144
        x1 = self.layer1(inputs)
        x1 = self.layer2(x1)
        # 72x72
        x2 = self.pool1(x1)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        # 36x36
        x3 = self.pool2(x2)
        x3 = self.layer5(x3)
        x3 = self.layer6(x3)
        # 18x18
        x4 = self.pool3(x3)
        x4 = self.layer7(x4)
        x4 = self.layer8(x4)
        # 9x9
        x5 = self.pool4(x4)
        x5 = self.layer9(x5)
        x5 = self.layer10(x5)

        # decoder
        x6 = self.conv1(x5)
        # 18x18
        x6 = self.up1(x6)
        x6 = self.bn1(x6, training=training)
        x6 = self.relu1(x6)
        x6 = tf.keras.layers.concatenate([x4,x6], axis=3)
        x6 = self.layer11(x6)
        x6 = self.layer12(x6)
        x7 = self.conv2(x6)
        # 36x36
        x7 = self.up2(x7)
        x7 = self.bn2(x7, training=training)
        x7 = self.relu2(x7)
        x7 = tf.keras.layers.concatenate([x3,x7], axis=3)
        x7 = self.layer13(x7)
        x7 = self.layer14(x7)
        x8 = self.conv3(x7)
        # 72x72
        x8 = self.up3(x8)
        x8 = self.bn3(x8, training=training)
        x8 = self.relu3(x8)
        x8 = tf.keras.layers.concatenate([x2, x8], axis=3)
        x8 = self.layer15(x8)
        x8 = self.layer16(x8)
        x9 = self.conv4(x8)
        # 144x144
        x9 = self.up4(x9)
        x9 = self.bn4(x9, training=training)
        x9 = self.relu4(x9)
        x9 = tf.keras.layers.concatenate([x1, x9], axis=3)
        x9 = self.layer17(x9)
        x9 = self.layer18(x9)
        x10 = self.conv5(x9)

        output = self.softmax(x10)
        # print(output.shape)

        return output