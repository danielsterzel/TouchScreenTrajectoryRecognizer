from tensorflow.keras import layers


class BiLSTM:
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape
