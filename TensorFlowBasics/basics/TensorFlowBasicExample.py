import tensorflow as tf
input_size = 2
output_size = 1


inputs = tf.placeholder(tf.float32, [None, input_size])
targets = tf.placeholder(tf.float32, [None, output_size])


weigths_random = tf.random_uniform([input_size, output_size], minval=-0.1, maxval=0.1)
bias_random = tf.random_uniform([output_size], minval=-0.1, maxval=0.1)

weights = tf.Variable(weigths_random)
biases =  tf.Variable(bias_random)
'''wx + b = y'''
outputs = tf.matmul(inputs, weights) + biases

mean_square_error = tf.losses.mean_squared_error(labels=targets, predictions=outputs)

optimize = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(mean_square_error)