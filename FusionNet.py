from Utils import *


class FusionNet(tf.keras.layers.Layer):
    def __init__(self, class_num):
        super(FusionNet, self).__init__()
        self.layer1 = ConvBlock(filter_num=64)
        self.layer2 = ResBlock(filter_num=64)
        self.layer3 = ConvBlock(filter_num=64)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer4 = ConvBlock(filter_num=128)
        self.layer5 = ResBlock(filter_num=128)
        self.layer6 = ConvBlock(filter_num=128)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer7 = ConvBlock(filter_num=256)
        self.layer8 = ResBlock(filter_num=256)
        self.layer9 = ConvBlock(filter_num=256)
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer10 = ConvBlock(filter_num=512)
        self.layer11 = ResBlock(filter_num=512)
        self.layer12 = ConvBlock(filter_num=512)
        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.layer13 = ConvBlock(filter_num=1024)
        self.layer14 = ResBlock(filter_num=1024)
        self.layer15 = ConvBlock(filter_num=1024)
        self.deconv1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')
        self.layer16 = ConvBlock(filter_num=512)
        self.layer17 = ResBlock(filter_num=512)
        self.layer18 = ConvBlock(filter_num=512)
        self.deconv2 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')
        self.layer19 = ConvBlock(filter_num=256)
        self.layer20 = ResBlock(filter_num=256)
        self.layer21 = ConvBlock(filter_num=256)
        self.deconv3 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')
        self.layer22 = ConvBlock(filter_num=128)
        self.layer23 = ResBlock(filter_num=128)
        self.layer24 = ConvBlock(filter_num=128)
        self.deconv4 = tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_normal')
        self.layer25 = ConvBlock(filter_num=64)
        self.layer26 = ResBlock(filter_num=64)
        self.layer27 = ConvBlock(filter_num=64)
        self.conv = tf.keras.layers.Conv2D(class_num, (3, 3), padding='same', kernel_initializer='he_normal')
        self.softmax = tf.keras.layers.Activation(activation='softmax')

    def call(self, inputs):
        # encoder
        # 144x144
        x1 = self.layer1(inputs)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        # 72x72
        x2 = self.pool1(x1)
        x2 = self.layer4(x2)
        x2 = self.layer5(x2)
        x2 = self.layer6(x2)
        # 36x36
        x3 = self.pool2(x2)
        x3 = self.layer7(x3)
        x3 = self.layer8(x3)
        x3 = self.layer9(x3)
        # 18x18
        x4 = self.pool3(x3)
        x4 = self.layer10(x4)
        x4 = self.layer11(x4)
        x4 = self.layer12(x4)
        # 9x9
        x5 = self.pool4(x4)

        # bridge
        x5 = self.layer13(x5)
        x5 = self.layer14(x5)
        x5 = self.layer15(x5)

        # decoder
        # 18x18
        x6 = self.deconv1(x5)
        x6 = tf.keras.layers.add([x6, x4])
        x6 = self.layer16(x6)
        x6 = self.layer17(x6)
        x6 = self.layer18(x6)
        # 36x36
        x7 = self.deconv2(x6)
        x7 = tf.keras.layers.add([x7, x3])
        x7 = self.layer19(x7)
        x7 = self.layer20(x7)
        x7 = self.layer21(x7)
        # 72x72
        x8 = self.deconv3(x7)
        x8 = tf.keras.layers.add([x8, x2])
        x8 = self.layer22(x8)
        x8 = self.layer23(x8)
        x8 = self.layer24(x8)
        # 144x144
        x9 = self.deconv4(x8)
        x9 = tf.keras.layers.add([x9, x1])
        x9 = self.layer25(x9)
        x9 = self.layer26(x9)
        x9 = self.layer27(x9)

        x10 = self.conv(x9)

        output = self.softmax(x10)
        # print(output.shape)

        return output