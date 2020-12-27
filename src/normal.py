import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_normal_df():
    return pd.DataFrame({
        'Normal': np.random.normal(size=100, loc=50),
        'Generated': [sum([np.random.rand() for j in range(0, 100)]) for i in range(0, 100)]
    })


if __name__ == '__main__':
    normal = load_normal_df()
    sns.displot(normal, kind="kde", fill=True)
    plt.show()
