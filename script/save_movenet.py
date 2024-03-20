import os
import tensorflow as tf
import tensorflow_hub as hub

os.environ["TFHUB_CACHE_DIR"] = "./dist"
module = hub.resolve("https://tfhub.dev/google/movenet/singlepose/lightning/4")
print(module)

if not os.path.exists("dist/movenet-lightning"):
    os.rename(module, "dist/movenet-lightning")

model = tf.saved_model.load("dist/movenet-lightning")
print(model)
