# %% loading modules
import matplotlib.pyplot as plt
import numpy as np

# %% load data
first_pic = np.loadtxt("pair_compare/incr_1_layer.csv", delimiter=",")
second_pic = np.loadtxt("pair_compare/incr_2_layer.csv", delimiter=",")

# let's plot both first
plt.plot(first_pic, label="first")
plt.plot(second_pic, label="second")
plt.xlabel("Channels")
plt.ylabel("Trend of both images")
plt.legend()
plt.show()

# do some simple processing first
difference = first_pic - second_pic
difference = np.abs(difference)
plt.plot(difference, label="difference")
plt.xlabel("Channels")
plt.ylabel("Absolute difference")
plt.legend()
plt.show()

# this is in no way concrete but we have something more.
ratio_diff = difference / np.abs(first_pic + second_pic)
plt.plot(ratio_diff, label="difference ratio")
plt.xlabel("Channels")
plt.ylabel("Ratio of differnce")
plt.legend()
plt.show()
