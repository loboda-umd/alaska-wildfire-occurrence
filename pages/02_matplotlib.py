import reacton
import solara
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.animation as animation


zoom = solara.reactive(2)
center = solara.reactive((20, 0))


class PauseAnimation:
    def __init__(self):
        self.fig = Figure()
        ax = self.fig.subplots()
        ax.set_title('Click to pause/resume the animation')
        x = np.linspace(-0.1, 0.1, 1000)

        # Start with a normal distribution
        self.n0 = (1.0 / ((4 * np.pi * 2e-4 * 0.1) ** 0.5)
                   * np.exp(-x ** 2 / (4 * 2e-4 * 0.1)))
        self.p, = ax.plot(x, self.n0)

        self.animation = animation.FuncAnimation(
            self.fig, self.update, frames=200, interval=50, blit=True)
        self.paused = False

        self.fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def update(self, i):
        self.n0 += i / 100 % 5
        self.p.set_ydata(self.n0 % 20)
        return (self.p,)

    def get_fig(self):
        return self.fig


@solara.component
def Page():
    # do this instead of plt.figure()
    #fig = Figure()
    #ax = fig.subplots()
    #ax.plot([1, 2, 3], [1, 4, 9])
    pa = PauseAnimation()
    return solara.FigureMatplotlib(pa.get_fig())
