import matplotlib.pyplot as plt
from matplotlib import animation


def plot_X_Y(df, x_name, y_name, y_value_name):
    df.plot(kind="scatter", x=x_name, y=y_name, alpha=0.1,
            c=y_value_name, cmap=plt.get_cmap("jet"))
    plt.show()


def plot_animation(frames, repeat=False, interval=40):
    print("Frames: {}".format(len(frames)))
    plt.close()
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat,
                                   interval=interval)


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,
