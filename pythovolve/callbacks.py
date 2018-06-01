class Callback:
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def on_generation_start(self):
        pass

    def on_generation_end(self):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass


class EarlyStopper(Callback):  # todo stop on no progress for n_steps
    def __init__(self, algorithm, max_generations=1000):
        super().__init__(algorithm)
        self.max_gen = max_generations

    def on_generation_end(self):
        if self.algorithm.generation >= self.max_gen:
            self.algorithm.stop_evolving = True
            print(f"Stopping after {self.algorithm.generation} generations")