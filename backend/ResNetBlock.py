from tensorflow.keras import layers

def conv_bn_relu(img, filter_count, kernel_size=3, stride=1):
    img = layers.Conv2D(filter_count, kernel_size, strides=stride, padding='same')(img)
    img = layers.BatchNormalization()(img)
    img = layers.ReLU()(img)
    return img

def build_residual_block(img, filter_count, stride=1):
    skip = img
    x = conv_bn_relu(img, filter_count, stride)
    x = layers.Conv2D(filter_count, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if stride !=1 or skip.shape[-1] != filter_count:
        skip = layers.Conv2D(filter_count, 1,strides=stride, padding='same')(skip)
        skip = layers.BatchNormalization()(skip)


    x = layers.Add()([x, skip])
    x = layers.ReLU()(x)
    return x
