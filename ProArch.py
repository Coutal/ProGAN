import tensorflow as tf
from keras import backend as K
from keras.engine import input_layer
from keras.layers import Dense, Conv2D, LeakyReLU, UpSampling2D
from keras import Model


def pixel_norm(inputs):
    inputs_value = tf.unstack(inputs, axis=[0, 1])
    C = len(inputs_value[0][0])
    for x in range(len(inputs_value)):
        for y in range(len(inputs_value[0])):
            s = (sum(inputs_value[x][y]) / C) + K.epsilon()
            inputs_value[x][y] = inputs_value[x][y] * s
    return tf.convert_to_tensor(inputs_value)


pixelNorm = tf.keras.layers.Lambda(pixel_norm)


class ProGan():
    net = {}

    def __init__(self):
        # self.net['input'] = keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        self.net['input'] = input_layer.Input(shape=(512,1,1,))
        lvl1 = []
        lvl1.append(pixelNorm(self.net['input']))
        lvl1.append( Conv2D(512, (4,4), activation="LeakyReLU") )
        self.net['lvl1']

        model = Model()


    # def make_lvl(self, in_shape, out_shape, input):



class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList
        from pro_gan_pytorch.CustomLayers import GenGeneralConvBlock, GenInitialBlock
        from torch.nn.functional import interpolate

        super(Generator, self).__init__()
