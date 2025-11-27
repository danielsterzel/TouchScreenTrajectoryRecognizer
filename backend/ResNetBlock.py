from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

def conv_bn_relu(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def conv_bn(x, filters, kernel_size=3, stride=1):
    x = layers.Conv2D(filters,kernel_size, strides=stride, padding='same', use_bias=False, kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    return x

def build_residual_block(x, filters, stride=1):
    skip = x

    out = conv_bn_relu(x, filters, kernel_size=3, stride=stride)
    out = conv_bn(out, filters, kernel_size=3, stride=1)

    if stride != 1 or int(skip.shape[-1]) != filters:
        skip = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(skip)
        skip = layers.BatchNormalization()(skip)

    out = layers.Add()([out, skip])
    out = layers.ReLU()(out)

    return out
