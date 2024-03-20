import onnxruntime as ort
import tensorflow as tf
import os
import sys
print(sys.path[0])
sys.path.append(os.getcwd())

from src.format import load_and_resize_image    # noqa: E402

image_np = tf.cast(load_and_resize_image('resource/p-001.jpg'), dtype=tf.int32).numpy()
print(image_np.shape)

ort_sess = ort.InferenceSession('dist/onnx/movenet-lightning.onnx')
outputs = ort_sess.run(None, {'input':image_np})

print(outputs)
print(outputs[0].shape)