import tensorflow as tf


class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filter_num, (3,3), padding='same', kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(activation='relu')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        output = self.relu(x)

        return output


# FusionNet
class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num):
        super(ResBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filter_num, (1, 1), padding='same', kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filter_num, (3, 3), padding='same', kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation(activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filter_num, (1, 1), padding='same', kernel_initializer='he_normal')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.Activation(activation='relu')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = tf.keras.layers.add([inputs, x])
        output = self.relu3(x)

        return output


# SegNet
class MaxPoolingWithArgmax2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if tf.keras.backend.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                tf.keras.backend.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = tf.keras.backend.cast(argmax, tf.keras.backend.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


class MaxUnpooling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with tf.compat.v1.variable_scope(self.name):
            mask = tf.keras.backend.cast(mask, "int32")
            input_shape = tf.shape(updates, out_type="int32")
            input_shape = [updates.shape[i] or input_shape[i] for i in range(4)]
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.size[0],
                    input_shape[2] * self.size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.keras.backend.ones_like(mask, dtype="int32")
            batch_shape = tf.keras.backend.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
            batch_range = tf.keras.backend.reshape(
                tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = tf.size(updates)
            indices = tf.keras.backend.transpose(tf.keras.backend.reshape(tf.keras.backend.stack([b, y, x, f]), [4, updates_size]))
            values = tf.keras.backend.reshape(updates, [updates_size])
            ret = tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.size[0],
            mask_shape[2] * self.size[1],
            mask_shape[3],
        )


# DeepLabV3+
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation(activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding='same', kernel_initializer='he_normal')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.Activation(activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer='he_normal')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.Activation(activation='relu')

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride))
        self.downsample.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = tf.keras.layers.add([residual, x])

        output = self.relu3(x)

        return output


class ResNet(tf.keras.layers.Layer):
    def __init__(self, layer_params):
        super(ResNet, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', kernel_initializer='he_normal')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.Activation(activation='relu')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        self.layer1 = self._make_bottleneck_layer(filter_num=64, blocks=layer_params[0])
        self.layer2 = self._make_bottleneck_layer(filter_num=128, blocks=layer_params[1], stride=2)
        self.layer3 = self._make_bottleneck_layer(filter_num=256, blocks=layer_params[2], stride=2)
        self.layer4 = self._make_bottleneck_layer(filter_num=512, blocks=layer_params[3])

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()

    def _make_bottleneck_layer(self, filter_num, blocks, stride=1):
        res_block = tf.keras.Sequential()
        res_block.add(BottleNeck(filter_num, stride=stride))

        for _ in range(1, blocks):
            res_block.add(BottleNeck(filter_num, stride=1))

        return res_block

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.pool1(x)
        x1 = self.layer1(x, training=training)
        x2 = self.layer2(x1, training=training)
        x3 = self.layer3(x2, training=training)
        x4 = self.layer4(x3, training=training)
        output = self.avgpool(x4)

        return output, x1, x4


class ConvBlock2(tf.keras.layers.Layer):
    def __init__(self, filter_num, kernel_size, dilation_rate, use_bias):
        super(ConvBlock2, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filter_num, kernel_size, dilation_rate=dilation_rate, padding='same', use_bias=use_bias, kernel_initializer='he_normal')
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation(activation='relu')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        output = self.relu(x)

        return output


class ASPP(tf.keras.layers.Layer):
    def __init__(self):
        super(ASPP, self).__init__()
        self.conv1 = ConvBlock2(filter_num=256, kernel_size=(1, 1), dilation_rate=1, use_bias=True)
        self.out1 = ConvBlock2(filter_num=256, kernel_size=(1, 1), dilation_rate=1, use_bias=False)
        self.out6 = ConvBlock2(filter_num=256, kernel_size=(3, 3), dilation_rate=6, use_bias=False)
        self.out12 = ConvBlock2(filter_num=256, kernel_size=(3, 3), dilation_rate=12, use_bias=False)
        self.out18 = ConvBlock2(filter_num=256, kernel_size=(3, 3), dilation_rate=18, use_bias=False)
        self.conv2 = ConvBlock2(filter_num=256, kernel_size=(1, 1), dilation_rate=1, use_bias=False)

    def call(self, inputs, training=False):
        dims = inputs.shape

        x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(inputs)
        x = self.conv1(x)

        x = tf.keras.layers.UpSampling2D(size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear")(x)

        x1 = self.out1(inputs)
        x6 = self.out6(inputs)
        x12 = self.out12(inputs)
        x18 = self.out18(inputs)

        x = tf.keras.layers.concatenate([x, x1, x6, x12, x18], axis=-1)
        output = self.conv2(x)

        return output