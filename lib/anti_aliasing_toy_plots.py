#cell 1
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

#cell 2
y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
plt.scatter(x, y)
plt.plot(x, y, '--o')

y = [0, 1, 0, 1, 0, 1, 0]
x = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5]
plt.scatter(x, y)
plt.plot(x, y, '--r')

plt.xlabel('Spatial position', fontsize=12)
plt.ylabel('Signal value', fontsize=12)
plt.xlim(0, 15)

#Cell 3
y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0]
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
plt.scatter(x, y)
plt.plot(x, y, '--o')

y = [1, 1, 1, 1, 1, 1, 1, 1]
x = [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5, 14.5]
plt.scatter(x, y)
plt.plot(x, y, '--g')

plt.xlabel('Spatial position', fontsize=12)
plt.ylabel('Signal value', fontsize=12)
plt.xlim(0, 15)