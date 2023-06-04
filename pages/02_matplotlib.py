import reacton
import solara as sol
from matplotlib.figure import Figure

zoom = solara.reactive(2)
center = solara.reactive((20, 0))


@solara.component
def Page():
    # do this instead of plt.figure()
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    return solara.FigureMatplotlib(fig)
