from multiprocessing import Queue
from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure


class ProgressPlot:
    def __init__(self, max_generations: int, data_queue: Queue, fig: Figure = None,
                 axes: Sequence[Axes] = None):

        # these imports can't be at the top of the file as they should only be
        # imported by the process running the plots. If both processes import
        # matplotlib, the child process will crash on MacOS.
        # see https://stackoverflow.com/questions/9879371
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.animation import FuncAnimation

        self.max_generations = max_generations
        # set to True if there's new data to plot, set to false after it has been plotted
        self.stale = False

        if fig is not None and axes is not None:
            self.fig = fig
            self.axes = axes
        else:
            self.fig, axes = plt.subplots()
            self.axes = [axes]

        self.data_queue = data_queue
        self.current_best_scores = []
        self.best_scores = []
        self.generation = 0

        self.total_line, = self.axes[0].plot([], [], 'r-', animated=True, label="Total best")
        self.current_line, = self.axes[0].plot([], [], 'g.', animated=True, label="Generation best")

        # setup the animation
        self.animation = FuncAnimation(self.fig, self._update, init_func=self._init,
                                       blit=True, interval=1000 // 20)

        self.legend = self.axes[0].legend()

        sns.set()
        plt.show()

    def _get_newest_data(self):
        while not self.data_queue.empty():
            current_best_score, best = self.data_queue.get()
            self.current_best_scores.append((self.generation, current_best_score))
            self.best_scores.append((self.generation, best.score))
            self.best = best
            self.generation += 1
            self.stale = True

    def _init(self):
        self._get_newest_data()

        if len(self.best_scores) > 0:
            x_max = min(self.max_generations - 1, int(len(self.best_scores) * 2 + 10))
            max_score = max(score[1] for score in self.best_scores)
            min_score = self.best.score
            y_min = min_score - (max_score - min_score) * 0.3
            y_max = max_score + (max_score - min_score) * 0.1
            y_max = y_max if not y_min == y_max else y_min + 1

        else:
            x_max = 100
            y_min = 0
            y_max = 1e-5

        ax = self.axes[0]
        ax.set_title("Progress")
        ax.set_xlim(0, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Score")

        self.legend = ax.legend()

        return self.current_line, self.total_line, self.legend

    def _update(self, _):
        self._get_newest_data()

        if len(self.best_scores) > 0 and self.stale:
            ax = self.axes[0]

            # update range of x-axis
            _, x_max = ax.get_xlim()
            if len(self.best_scores) + 1 > x_max * 0.95 and not x_max == self.max_generations - 1:
                ax.set_xlim(0, min(self.max_generations - 1, int(x_max * 2 + 10)))
                ax.figure.canvas.draw()

            # update range of y-axis
            y_min, y_max = ax.get_ylim()
            max_score = max(score[1] for score in self.best_scores)
            min_score = self.best.score
            if max_score > y_max or min_score < y_min:
                new_min = min_score - (max_score - min_score) * 0.3
                new_max = max_score + (max_score - min_score) * 0.1
                new_max = new_max if not new_min == new_max else new_min + 1
                ax.set_ylim(new_min, new_max)
                ax.figure.canvas.draw()

            self.current_line.set_data(zip(*sorted(self.current_best_scores)))
            self.total_line.set_data(zip(*sorted(self.best_scores)))

        return self.current_line, self.total_line, self.legend


class TSPPlot(ProgressPlot):
    def __init__(self, max_generations: int, problem, data_queue: Queue):

        import matplotlib.pyplot as plt

        self.problem = problem
        self.path_lines = []
        self.city_points = []

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        super().__init__(max_generations, data_queue, fig, axes)

    def _init(self):
        super()._init()
        ax = self.axes[1]
        area = self.problem.defined_area

        ax.set_title("Best Solution")
        ax.set_xlim(area.min.x, area.max.x)
        ax.set_ylim(area.min.y, area.max.y)

        for point in self.city_points:
            del point

        x_cities = [city.x for city in self.problem.cities]
        y_cities = [city.y for city in self.problem.cities]
        self.city_points = ax.plot(x_cities, y_cities, ls="", marker="*", label="Cities")

        self._plot_paths()

        return (self.current_line, self.total_line, self.legend, *self.path_lines)

    def _update(self, frame):
        super()._update(frame)
        self._plot_paths()
        return (self.current_line, self.total_line, self.legend, *self.path_lines)

    def _plot_paths(self):
        if not self.best_scores or not self.stale:
            return

        ax = self.axes[1]

        while self.path_lines:
            # to completely get rid of the lines, this is necessary
            # see https://stackoverflow.com/questions/4981815
            self.path_lines.pop(0).remove()

        path = [self.problem.cities[idx] for idx in self.best.phenotype]
        self.path_lines = ax.plot([path[-1].x, path[0].x], [path[-1].y, path[0].y], "k-")

        for start, dest in zip(path, path[1:]):
            self.path_lines += ax.plot([dest.x, start.x], [dest.y, start.y], "k-")
