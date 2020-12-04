import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == '__main__':
    df = pd.DataFrame({
        'Normal': np.random.normal(size=100, loc=50),
        'Generated': [sum([np.random.rand() for j in range(0, 100)]) for i in range(0, 100)]
    })

    sns.displot(df, kind="kde", fill=True)

    plt.show()
