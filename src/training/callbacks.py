from abc import ABC, abstractmethod


class Callback(ABC):

    def on_train_begin(self, trainer):
        pass

    def on_epoch_begin(self, trainer, epoch):
        pass

    def on_epoch_end(self, trainer, epoch, logs):
        pass

    def on_train_end(self, trainer):
        pass

class EarlyStopping(Callback):

    def __init__(
        self,
        monitor="val_loss",
        patience=3,
        min_delta=0.0,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta

        self.best = float("inf")
        self.wait = 0

    def on_epoch_end(self, trainer, epoch, logs):

        value = logs[self.monitor]

        if value < self.best - self.min_delta:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            trainer.stop_training = True

class ModelCheckpoint(Callback):

    def __init__(
        self,
        filepath,
        monitor="val_loss",
    ):

        self.filepath = filepath
        self.monitor = monitor
        self.best = float("inf")

    def on_epoch_end(
        self,
        trainer,
        epoch,
        logs,
    ):

        metric = logs[self.monitor]

        if metric < self.best:

            self.best = metric

            trainer.model.save_weights(
                self.filepath
            )