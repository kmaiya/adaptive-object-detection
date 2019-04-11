# %% loading modules
import matplotlib.pyplot as plt
import numpy as np

# %% load data
first_pic = np.loadtxt("pair_compare/1.csv", delimiter=",")
second_pic = np.loadtxt("pair_compare/2.csv", delimiter=",")

# let's plot both first
plt.plot(first_pic, label="first")
plt.plot(second_pic, label="second")
plt.legend()
plt.show()

# do some simple processing first
difference = first_pic - second_pic
difference = np.abs(difference)
plt.plot(difference, label="difference")
plt.legend()
plt.show()

# this is in no way concrete but we have something more.
ratio_diff = difference / np.abs(first_pic + second_pic)
plt.plot(ratio_diff, label="difference ratio")
plt.xlabel("Channels")
plt.ylabel("Ratio of differnce")
plt.legend()
plt.show()
