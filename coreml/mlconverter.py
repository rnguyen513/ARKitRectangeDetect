import coremltools
import tensorflow as tf

model_path = "/Users/iseelab/Downloads/cnn_single_epoch_0.h5"

keras_model = tf.keras.models.load_model(model_path)

model = coremltools.convert(keras_model, convert_to="mlprogram")

model.save("converted_single_epoch")
