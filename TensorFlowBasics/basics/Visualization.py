import numpy as np
import matplotlib.pyplot as plt
import math

x = np.arange(-5,5,.1)

y = [math.atan(aux) for aux in x]

print(y)
print(x)


plt.plot(x,y)
plt.title('ATAN')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



colors =  np.arange(0,100).reshape(10,10)
plt.imshow(colors)
plt.colorbar()
plt.show()