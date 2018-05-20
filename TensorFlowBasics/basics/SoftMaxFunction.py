import matplotlib.pyplot as plt
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x) , axis=0)

x = np.arange(0, 4, 0.1) 
print('Rango dibujo: ' , x)

ones2 = [0.2 * np.ones_like(x)]
print(ones2)

ones4 = .4*np.ones_like(x)
print(ones4)

ones = np.ones_like(x)
print(ones)

scores = np.vstack([x, ones, ones2, ones4])

plt.plot(x, softmax(scores).T)
plt.show()
