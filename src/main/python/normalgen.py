import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.random.normal(size=100, loc=50)
y = [sum([np.random.rand() for j in range(1, 100)]) for i in range(1, 100)]

sns.distplot(x)
sns.distplot(y)

plt.show()
