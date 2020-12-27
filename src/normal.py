import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_normal_df():
    return pd.DataFrame({
        'Normal': np.random.normal(size=100, loc=50),
        'Generated': [sum([np.random.rand() for j in range(0, 100)]) for i in range(0, 100)]
    })


if __name__ == '__main__':
    df = load_normal_df()
    df.plot(kind="kde")
    plt.show()
