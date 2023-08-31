import numpy as np
from perceptron import Perceptron
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df["target"] = iris.target
X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 0, -1, 1)
percepton = Perceptron(eta=0.1, n_iter=10)

percepton.fit(X, y)
plt.plot(range(1, len(percepton.errors) + 1), percepton.errors, marker="o")
plt.xlabel("Epochs")
plt.ylabel("Number of updates")
plt.show()
