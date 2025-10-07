####################################
# Script for easy live plotting
# made by Jacob Molnia
####################################

from collections import defaultdict

import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self, title="Training Progress"):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.fig.suptitle(title)
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)

        self.lines = {}
        self.data = defaultdict(list)
        self.steps = defaultdict(list)

    def update(self, metric_name, value, step=None):
        if step is None:
            step = len(self.data[metric_name])

        self.data[metric_name].append(value)
        self.steps[metric_name].append(step)
        if metric_name not in self.lines:
            (line,) = self.ax.plot([], [], label=metric_name)
            self.lines[metric_name] = line
            self.ax.legend()

        self.lines[metric_name].set_data(self.steps[metric_name], self.data[metric_name])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)  # Needed for the window to update

    def final_save(self, filename="training_plot.png"):
        plt.ioff()  # Turn off interactive mode for final save
        self.fig.savefig(filename)

    def close(self):
        plt.close(self.fig)
