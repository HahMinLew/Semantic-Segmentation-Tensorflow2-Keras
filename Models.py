from FCN8s import *
from UNet import *
from SegNet import *
from SegNet_basic import *
from FusionNet import *
from DeepLabV3Plus import *


class SSM(tf.keras.Model):
    def __init__(self, class_num, model_name):
        super(SSM, self).__init__(name="Semantic_Segmentation_Model_{}".format(model_name))
        if model_name == 'fcn8s':
            self.model = FCN8s(class_num=class_num)
        elif model_name == 'unet':
            self.model = UNet(class_num=class_num)
        elif model_name == 'segnet':
            self.model = SegNet(class_num=class_num)
        elif model_name == 'segnetbasic':
            self.model = SegNetBasic(class_num=class_num)
        elif model_name == 'fusionnet':
            self.model = FusionNet(class_num=class_num)
        elif model_name == 'deeplabv3plus':
            self.model = DeepLabV3Plus(class_num=class_num)

    def call(self, inputs):
        output = self.model(inputs)

        return output


class_num = 4 # The number of classes you want to segment


def fcn8s():
    return SSM(class_num=class_num, model_name='fcn8s')


def unet():
    return SSM(class_num=class_num, model_name='unet')


def segnet():
    return SSM(class_num=class_num, model_name='segnet')


def segnetbasic():
    return SSM(class_num=class_num, model_name='segnetbasic')


def fusionnet():
    return SSM(class_num=class_num, model_name='fusionnet')


def deeplabv3plus():
    return SSM(class_num=class_num, model_name='deeplabv3plus')


# ch = 3
# model = deeplabv3plus()
# model.build(input_shape=(None, 144, 144, ch))
# model.summary()