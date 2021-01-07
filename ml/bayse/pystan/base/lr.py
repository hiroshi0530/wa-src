import numpy as np
import pandas as pd
import pystan
import matplotlib.pyplot as plt
import seaborn as sns

X = np.array([i * 2 for i in range(1,21)])
np.random.seed(1000)
epsilon = np.random.normal(0,5,20)
Y = 2 * X + epsilon
plt.scatter(X, Y)
plt.show()
