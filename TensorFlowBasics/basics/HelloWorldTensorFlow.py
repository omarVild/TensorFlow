import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello ")
world = tf.constant("world")

zeros =tf.zeros((4,4))

helloPH = tf.placeholder(tf.string, name='hello')
namePH = tf.placeholder(tf.string, name='name')

addOpr = tf.add(helloPH, namePH)


with tf.Session() as sess:
    res = sess.run(hello+world)
    print(res)
    res2 = sess.run(addOpr, feed_dict={helloPH:'Hello ', namePH:'Omar'}  )
    print(res2)

    


