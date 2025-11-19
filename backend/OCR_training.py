# from tensorflow.keras import layers
# import tensorflow as tf
# import tensorflow.keras.backend as K
# #backend because that is where the loss functions are.
#
# from OCR import OCR
#
# # ctc loss needs inputs, labels, input_len and label_len
# class OCRTraining:
#     def __init__(self, input_shape=(64,64,1), num_classes=100):
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.base_model = OCR(input_shape).build_model(num_classes)
#         self.ctc_model = None
#     def build_ctc_model(self):
#
#         labels = layers.Input(name='labels', shape=(None, ), dtype='int32')
#         input_len = layers.Input(name='input_len', shape=(1,), dtype='int32')
#         label_len = layers.Input(name='label_len', shape=(1,), dtype='int32')
#
#         y_pred = self.base_model.output
#
#         loss = layers.Lambda(lambda args : K.ctc_batch_cost(*args),name='ctc_loss') ([labels,y_pred, input_len, label_len])
#
#         self.ctc_model = tf.keras.Model(
#             inputs=[self.base_model.input,
#                     labels,
#                     input_len,
#                     label_len
#                     ],
#             outputs=loss,
#         )
#
#         return self.ctc_model