import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello ")
world = tf.constant("world")

zeros =tf.zeros((4,4))


with tf.Session() as sess:
    res = sess.run(hello+world)

    
print(res)