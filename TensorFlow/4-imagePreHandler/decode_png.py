# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import tensorflow as tf

image_raw = tf.gfile.FastGFile("cat.png", "rb").read()
img_data = tf.image.decode_jpeg(image_raw)
img_data = tf.image.decode_png(image_raw)

with tf.Session() as sess:
    print(sess.run(img_data))
