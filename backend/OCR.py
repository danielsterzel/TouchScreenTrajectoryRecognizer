from tensorflow.keras import layers, models
import ResNetBlock as resblock
# import BiLSTM as bilstm

class OCR:
    def __init__(self,  input_shape=(64,64,1)):
        self.model = None
        self.input_shape = input_shape
        self.output = None

    def build_model(self, num_classes):
        inputs = layers.Input(shape=self.input_shape)
        # ----------- ResNet18 -----------
        x = resblock.conv_bn_relu(inputs, 64, 3, 1) # (64,64, 64)
        x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x) # (32, 32, 64)

        # residual block 1
        x = resblock.build_residual_block(x, 64) # (32, 32, 64)
        x = resblock.build_residual_block(x, 64) # (32, 32, 64)

        # residual block 2
        x = resblock.build_residual_block(x, 128, stride=2) # (16, 16, 128)
        x = resblock.build_residual_block(x, 128) # (16, 16, 128)

        # residual block 3
        x = resblock.build_residual_block(x, 256, stride=2) # (8, 8, 256)
        x = resblock.build_residual_block(x, 256) # (8, 8, 256)

        #residual block 4
        x = resblock.build_residual_block(x, 512, stride=2) # (4, 4, 512)
        x = resblock.build_residual_block(x, 512) # (4, 4, 512) -> 4x4 feature map

        # # ----------- BiLSTM -----------
        # x = layers.Reshape((4, 512 * 4)) (x)
        # x = bilstm.apply_bi_lstm(x)

        # outputs = layers.Dense(num_classes)(x)
        # self.output = outputs
        #
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs, outputs)
        self.model = model
        return model