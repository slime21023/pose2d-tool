import tensorflow as tf


def load_and_resize_image(image_path) -> tf.Tensor:
    # Load the input image.
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)
    return input_image
