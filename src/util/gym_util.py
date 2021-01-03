import matplotlib.pyplot as plt
from matplotlib import animation


def plot_animation(frames, repeat=False, interval=40):
    plt.close()
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat,
                                   interval=interval)


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,
