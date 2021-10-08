import tensorflow as tf

from configuration import Config
from core.efficientdet import EfficientDet, PostProcessing


def print_model_summary(network):
    sample_inputs = tf.random.normal(shape=(Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    sample_outputs = network(sample_inputs, training=True)
    network.summary()

model = EfficientDet()
print_model_summary(model)
load_weights_from_epoch = Config.load_weights_from_epoch
model.load_weights(filepath=Config.save_model_dir+"epoch-{}".format(load_weights_from_epoch))


dataset_list = tf.data.Dataset.list_files('./data/datasets/JPEGImages'+'\\*')

def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files('./data/datasets/JPEGImages'+'\\*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (512,512))
    # image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
# Model has only one input so each data point has one element
    yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.experimental_new_quantizer = True
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()
