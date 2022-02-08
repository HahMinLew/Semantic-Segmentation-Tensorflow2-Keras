from Utils import *


class SegNetBasic(tf.keras.layers.Layer):
    def __init__(self, class_num):
        super(SegNetBasic, self).__init__()
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
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')
        self.up1 = tf.keras.layers.UpSampling2D((2, 2))
        self.crop1 = tf.keras.layers.Cropping2D(cropping=((0, 1), (0, 1))) # According to your image size, you can adjust this code line
        self.layer14 = ConvBlock(filter_num=512)
        self.layer15 = ConvBlock(filter_num=512)
        self.layer16 = ConvBlock(filter_num=512)
        self.up2 = tf.keras.layers.UpSampling2D((2, 2))
        self.layer17 = ConvBlock(filter_num=512)
        self.layer18 = ConvBlock(filter_num=512)
        self.layer19 = ConvBlock(filter_num=512)
        self.up3 = tf.keras.layers.UpSampling2D((2, 2))
        self.layer20 = ConvBlock(filter_num=256)
        self.layer21 = ConvBlock(filter_num=256)
        self.layer22 = ConvBlock(filter_num=256)
        self.up4 = tf.keras.layers.UpSampling2D((2, 2))
        self.layer23 = ConvBlock(filter_num=128)
        self.layer24 = ConvBlock(filter_num=128)
        self.up5 = tf.keras.layers.UpSampling2D((2, 2))
        self.layer25 = ConvBlock(filter_num=64)
        self.layer26 = ConvBlock(filter_num=64)
        self.conv = tf.keras.layers.Conv2D(class_num, (3, 3), padding='same', kernel_initializer='he_normal')
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
        x4 = self.layer8(x3)
        x4 = self.layer9(x4)
        x4 = self.layer10(x4)
        # 9x9
        x4 = self.pool4(x4)
        x5 = self.layer11(x4)
        x5 = self.layer12(x5)
        x5 = self.layer13(x5)
        # 5x5
        x5 = self.pool5(x5)

        # decoder
        # 10x10
        x6 = self.up1(x5)
        # 9x9
        x6 = self.crop1(x6) # According to your image size, you can adjust this code line
        x6 = self.layer14(x6)
        x6 = self.layer15(x6)
        x6 = self.layer16(x6)
        # 18x18
        x7 = self.up2(x6)
        x7 = self.layer17(x7)
        x7 = self.layer18(x7)
        x7 = self.layer19(x7)
        # 36x36
        x8 = self.up3(x7)
        x8 = self.layer20(x8)
        x8 = self.layer21(x8)
        x8 = self.layer22(x8)
        # 72x72
        x9 = self.up4(x8)
        x9 = self.layer23(x9)
        x9 = self.layer24(x9)
        # 144x144
        x10 = self.up5(x9)
        x10 = self.layer25(x10)
        x10 = self.layer26(x10)
        x11 = self.conv(x10)

        output = self.softmax(x11)
        print(output.shape)

        return output
