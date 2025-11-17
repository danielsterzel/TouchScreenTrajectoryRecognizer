from tensorflow.keras import layers
import tensorflow as tf
from OCR import OCR

# ctc loss needs inputs, labels, input_len and label_len

class OCRTraining(OCR):
    def __init__(self, input_data, labels, input_shape = (64,64, 1)):
        super().__init__(input_shape=input_shape)
        self.ctc_model = None
