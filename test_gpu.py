import tensorflow as tf
print('TensorFlow版本:', tf.__version__)
print('可用GPU设备:', tf.config.list_physical_devices('GPU'))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print('GPU可用！')
else:
    print('未检测到可用GPU。')