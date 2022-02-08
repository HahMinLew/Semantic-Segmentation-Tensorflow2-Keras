from Utils import *


class DeepLabV3Plus(tf.keras.layers.Layer):
    def __init__(self, class_num):
        super(DeepLabV3Plus, self).__init__()
        self.resnet50 = ResNet(layer_params=[3, 4, 6, 3]) # resnet101 [3, 4, 23, 3]
        self.aspp = ASPP()
        self.conv1 = ConvBlock2(filter_num=48, kernel_size=(1, 1), dilation_rate=1, use_bias=False)
        self.conv2 = ConvBlock2(filter_num=256, kernel_size=(1, 1), dilation_rate=1, use_bias=False)
        self.conv3 = ConvBlock2(filter_num=256, kernel_size=(1, 1), dilation_rate=1, use_bias=False)
        self.conv4 = tf.keras.layers.Conv2D(class_num, (1, 1), padding='same', kernel_initializer='he_normal')
        self.softmax = tf.keras.layers.Activation(activation='softmax')

    def call(self, inputs, training=False):
        dims = inputs.shape

        x, x1, x4 = self.resnet50(inputs)

        x = self.aspp(x4)

        input_a = tf.keras.layers.UpSampling2D(size=(dims[1] // 4 // x.shape[1], dims[1] // 4 // x.shape[2]), interpolation='bilinear')(x)

        input_b = self.conv1(x1)

        x = tf.keras.layers.concatenate([input_a, input_b], axis=-1)
        x = self.conv2(x)
        x = self.conv3(x)

        x = tf.keras.layers.UpSampling2D(size=(dims[1] // x.shape[1], dims[1] // x.shape[2]), interpolation='bilinear')(x)

        x = self.conv4(x)
        output = self.softmax(x)

        return output