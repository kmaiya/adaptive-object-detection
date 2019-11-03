'''
Jupyter notebook code for generating object confidence standard devaition plots for
antialiased YOLO vs normal YOLO. 
'''

import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

a = [0.017, 0.014, 0.013, 0.012, 0.013, 0.009, 0.008, 0.007, 0.014, 0.014, 0.009, 0.014, 0.010, 0.012, 0.009, 0.013, 0.014, 0.018, 0.010, 0.012]
n = [0.026, 0.017, 0.019, 0.025, 0.026, 0.022, 0.020, 0.019, 0.022, 0.021, 0.018, 0.026, 0.022, 0.015, 0.022, 0.014, 0.022, 0.022, 0.014, 0.014]

plt.plot(a)
plt.plot(n)
plt.xlabel('Image frames', fontsize=12)
plt.ylabel('Object confidence deviation', fontsize=12)
plt.legend(['Anti-aliased', 'Normal'], loc='best')
plt.xticks(np.arange(20), ('5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100'))