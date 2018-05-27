import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


print(tf.__version__)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


num =55000
x_train = mnist.train.images[:num,:]
y_train = mnist.train.labels[:num,:]
        
def showNumber(image, label_tmp):
    imageData = image.reshape([28,28])
    plt.title('Nuumero: %d' % (label_tmp))
    plt.imshow(imageData, cmap=plt.get_cmap('gray_r'))
    plt.show()


x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
L = tf.placeholder(tf.float32, [None, 10])


cross_entropy = tf.reduce_mean(-tf.reduce_sum(L * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy )
sess = tf.InteractiveSession()


tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ls = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, L: batch_ls})


test_images = mnist.test.images[:num,:]
test_labels = mnist.test.labels[:num,:]

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
x_consec = np.arange(0,10,1)
for aux in range(20,30): 
    label_tmp= (test_labels[aux].argmax(axis=0))
    print("Imagen a imprimir: %s" % label_tmp )
    test_image =  np.array( [ test_images[aux]], dtype=float )
    resul = sess.run(y, feed_dict={x: test_image })
    plt.title('Probabilidades de ser el nuumero: %d' % (label_tmp))
    #grafica de probabilidades
    print('Probabilidades para cada nuumero:', resul)
    plt.plot(x_consec, resul[0])
    plt.show()
    showNumber(test_image,label_tmp)
    
    print(np.argmax(resul))
    print('---------------') 


