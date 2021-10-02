from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Dropout, MaxPooling2D, LeakyReLU, concatenate, BatchNormalization, Conv2DTranspose
from tensorflow.python.keras.models import Model
import tensorflow as tf
from model.common import normalize, denormalize, pixel_shuffle

# Network hyper-parameters:
kernel_size  =  4
filters_orig = 32
layer_depth  =  4
scale_factor= 4
# use_batch_norm = batch_size > 1

Upscale = lambda name: Lambda(
    lambda images: tf.image.resize(images, tf.shape(images)[-3:-1] * scale_factor),
    name=name)

ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]),
    # `images` is a tuple of 2 tensors.
    # We resize the first image tensor to the shape of the 2nd
    name=name)

def U_Net(layer_depth=4, filters_orig=32, kernel_size=4, batch_norm = True):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)
    x = Upscale(name='upscale_input')(x)
    x = unet(x, layer_depth, filters_orig, kernel_size, batch_norm, dropout=False)
    x = Lambda(denormalize)(x)
    return Model(x_in, x, name='unet')


def name_layer_factory(num=0, name_prefix="", name_suffix=""):
    """
    Helper function to name all our layers.
    """

    def name_layer_fn(layer):
        return '{}{}{}-{}'.format(name_prefix, layer, name_suffix, num)

    return name_layer_fn


def conv_bn_lrelu(filters, kernel_size=3, batch_norm=True,
                  kernel_initializer='he_normal', padding='same',
                  name_fn=lambda layer: "conv_bn_lrelu-{}".format(layer)):
    """
    Return a function behaving like a sequence convolution + BN + lReLU.
    :param filters:              Number of filters for the convolution
    :param kernel_size:          Kernel size for the convolutions
    :param batch_norm:           Flag to perform batch normalization
    :param kernel_initializer:   Name of kernel initialization method
    :param padding:              Name of padding option
    :param name_fn:              Function to name each layer of this sequence
    :return:                     Function chaining layers
    """

    def block(x):
        x = Conv2D(filters, kernel_size=kernel_size,
                   activation=None, kernel_initializer=kernel_initializer,
                   padding=padding, name=name_fn('conv'))(x)
        if batch_norm:
            x = BatchNormalization(name=name_fn('bn'))(x)
        x = LeakyReLU(alpha=0.3, name=name_fn('act'))(x)
        return x

    return block


def unet_conv_block(filters, kernel_size=3,
                    batch_norm=True, dropout=False,
                    name_prefix="enc_", name_suffix=0):
    """
    Return a function behaving like a U-Net convolution block.
    :param filters:              Number of filters for the convolution
    :param kernel_size:          Kernel size for the convolutions
    :param batch_norm:           Flag to perform batch normalization
    :param dropout:              NFlag to perform dropout between the two convs
    :param name_prefix:          Prefix for the layer names
    :param name_suffix:          FSuffix for the layer names
    :return:                     Function chaining layers
    """

    def block(x):
        # First convolution:
        name_fn = name_layer_factory(1, name_prefix, name_suffix)
        x = conv_bn_lrelu(filters, kernel_size=kernel_size, batch_norm=batch_norm,
                          name_fn=name_layer_factory(1, name_prefix, name_suffix))(x)
        if dropout:
            x = Dropout(0.2, name=name_fn('drop'))(x)

        # Second convolution:
        name_fn = name_layer_factory(2, name_prefix, name_suffix)
        x = conv_bn_lrelu(filters, kernel_size=kernel_size, batch_norm=batch_norm,
                          name_fn=name_layer_factory(2, name_prefix, name_suffix))(x)

        return x

    return block



def unet(x, layer_depth=4, filters_orig=32, kernel_size=4,
         batch_norm=True, dropout=True, final_activation='sigmoid'):
    """
    Define a U-Net network.
    :param x:                    Input tensor/placeholder
    :param filters_orig:         Number of filters for the 1st CNN layer
    :param kernel_size:          Kernel size for the convolutions
    :param batch_norm:           Flag to perform batch normalization
    :param dropout:              Flag to perform dropout
    :param final_activation:     Name of activation function for the final layer
    :return:                     Network (Keras Functional API)
    """
    num_channels = x.shape[-1]

    # Encoding layers: 32
    filters = filters_orig
    outputs_for_skip = []

    # layer_depth = 4
    for i in range(layer_depth):
        # Convolution block:
        x_conv = unet_conv_block(filters, kernel_size,
                                 dropout=dropout, batch_norm=batch_norm,
                                 name_prefix="enc_", name_suffix=i)(x)

        # We save the pointer to the output of this encoding block,
        # to pass it to its parallel decoding block afterwards:
        outputs_for_skip.append(x_conv)

        # Downsampling:
        x = MaxPooling2D(2)(x_conv)

        filters = min(filters * 2, 512)

    # Bottleneck layers:
    x = unet_conv_block(filters, kernel_size, dropout=dropout,
                        batch_norm=batch_norm, name_suffix='_btleneck')(x)

    # Decoding layers:
    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig)

        # Upsampling:
        name_fn = name_layer_factory(3, "ups_", i)
        x = Conv2DTranspose(filters, kernel_size=kernel_size, strides=2,
                            activation=None, kernel_initializer='he_normal',
                            padding='same', name=name_fn('convT'))(x)
        if batch_norm:
            x = BatchNormalization(name=name_fn('bn'))(x)
        x = LeakyReLU(alpha=0.3, name=name_fn('act'))(x)

        # Concatenation with the output of the corresponding encoding block:
        shortcut = outputs_for_skip[-(i + 1)]
        x = ResizeToSame(name='resize_to_same{}'.format(i))([x, shortcut])

        x = concatenate([x, shortcut], axis=-1, name='dec_conc{}'.format(i))

        # Convolution block:
        use_dropout = dropout and (i < (layer_depth - 2))
        x = unet_conv_block(filters, kernel_size,
                            batch_norm=batch_norm, dropout=use_dropout,
                            name_prefix="dec_", name_suffix=i)(x)

    # x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
    #            padding='same', name='dec_out1')(x)
    # x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu',
    #            padding='same', name='dec_out2')(x)
    x = Conv2D(filters=num_channels, kernel_size=1, activation=final_activation,
               padding='same', name='dec_output')(x)

    return x

#
# def edsr(scale, num_filters=64, num_res_blocks=8, res_block_scaling=None):
#     x_in = Input(shape=(None, None, 3))
#     x = Lambda(normalize)(x_in)
#
#     x = b = Conv2D(num_filters, 3, padding='same')(x)
#     for i in range(num_res_blocks):
#         b = res_block(b, num_filters, res_block_scaling)
#     b = Conv2D(num_filters, 3, padding='same')(b)
#     x = Add()([x, b])
#
#     x = upsample(x, scale, num_filters)
#     x = Conv2D(3, 3, padding='same')(x)
#
#     x = Lambda(denormalize)(x)
#     return Model(x_in, x, name="edsr")
