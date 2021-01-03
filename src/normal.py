import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def generate_normal():
    return pd.DataFrame({
        'Normal': np.random.normal(size=100, loc=50),
        'Generated': [sum([np.random.rand() for j in range(0, 100)]) for i in range(0, 100)]
    })


def plot_dist(dist):
    dist.plot(kind="kde")
    plt.show()


if __name__ == '__main__':
    normal = generate_normal()
    plot_dist(normal)
