from tensorflow.keras import layers, models
import ResNetBlock as resblock


class ResNet:
    def __init__(self,  input_shape=(64,64,1)):
        self.model = None
        self.input_shape = input_shape

    def build_model(self, num_classes):
        inputs = layers.Input(shape=self.input_shape)

        x = resblock.conv_bn_relu(inputs, 64, 3, 1)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

        # residual block 1
        x = resblock.build_residual_block(x, 64)
        x = resblock.build_residual_block(x, 64)

        # residual block 2
        x = resblock.build_residual_block(x, 128, stride=2)
        x = resblock.build_residual_block(x, 128)

        # residual block 3
        x = resblock.build_residual_block(x, 256, stride=2)
        x = resblock.build_residual_block(x, 256)

        #residual block 4
        x = resblock.build_residual_block(x, 512, stride=2)
        x = resblock.build_residual_block(x, 512)

        # output
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)

        model = models.Model(inputs, outputs)
        self.model = model