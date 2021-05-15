import matplotlib.pyplot as plt


def plot_X_Y(df, x_name, y_name, y_value_name):
    df.plot(kind="scatter", x=x_name, y=y_name, alpha=0.1,
            c=y_value_name, cmap=plt.get_cmap("jet"))
    plt.show()
