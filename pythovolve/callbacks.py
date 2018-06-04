import time
from typing import Optional

from pythovolve.individuals import Individual


class Callback:
    def __init__(self):
        self.algorithm = None

    def subscribe(self, algorithm):
        self.algorithm = algorithm

    def on_generation_start(self):
        pass

    def on_generation_end(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass


class TimerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.start_time = None

    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        run_time = time.time() - self.start_time
        try:
            print(f"Timer: Algorithm took {run_time:.2f} seconds and "
                  f"{run_time/self.algorithm.generation:.2f} seconds per generation.")
        except ZeroDivisionError:
            print(f"Timer: Algorithm took {run_time:.2f} seconds")


class ProgressLoggerCallback(Callback):
    def __init__(self, print_every: int = 10):
        super().__init__()
        self.print_every = print_every

    def on_generation_end(self):
        if self.algorithm.generation % self.print_every == 0:
            message = f"Progress: Generation {self.algorithm.generation} of {self.algorithm.max_generations} - " \
                      f"total best: {self.algorithm.best.score:.2f}"

            if self.algorithm.num_elites == 0:
                message += f", generation best: {self.algorithm.current_best.score:.2f}"

            print(message)


class EarlyStopCallback(Callback):
    def __init__(self, min_sigma: Optional[float] = None, max_no_progress: Optional[int] = None):
        super().__init__()
        self.max_no_progress = max_no_progress
        self.no_progress_cnt = 0
        self.min_sigma = min_sigma
        self.best: Individual = None

    def on_generation_start(self):
        self.best = self.algorithm.best

    def on_generation_end(self):
        if self.algorithm.best is not self.best:
            self.no_progress_cnt = 0
        else:
            self.no_progress_cnt += 1

        if self.min_sigma and hasattr(self.algorithm, "sigma") and self.algorithm.sigma < self.min_sigma:
            self.algorithm.stop_evolving = True
            print(f"EarlyStop: Stopping after {self.algorithm.generation} generations.")
            print(f"EarlyStop: Sigma is below threshold: {self.algorithm.sigma} < {self.min_sigma}")

        if self.max_no_progress and self.no_progress_cnt >= self.max_no_progress:
            self.algorithm.stop_evolving = True
            print(f"EarlyStop: Stopping after {self.algorithm.generation} generations.")
            print(f"EarlyStop: No progress for {self.no_progress_cnt} generations.")
