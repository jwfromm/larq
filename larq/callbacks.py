from typing import Any, Callable, MutableMapping, Optional

from tensorflow import keras


class HyperparameterScheduler(keras.callbacks.Callback):
    """Generic hyperparameter scheduler.

    !!! example
        ```python
        bop = lq.optimizers.Bop(threshold=1e-6, gamma=1e-3)
        adam = keras.optimizers.Adam(0.01)
        optimizer = lq.optimizers.CaseOptimizer(
            (lq.optimizers.Bop.is_binary_variable, bop), default_optimizer=adam,
        )
        callbacks = [
            HyperparameterScheduler(lambda x: 0.001 * (0.1 ** (x // 30)), "gamma", bop)
        ]
        ```
    # Arguments
    optimizer: the optimizer that contains the hyperparameter that will be scheduled.
        Defaults to `self.model.optimizer` if `optimizer == None`.
    schedule: a function that takes an epoch index as input
        (integer, indexed from 0) and returns a new hyperparameter as output.
    hyperparameter: str. the name of the hyperparameter to be scheduled.
    update_freq: str (optional), denotes on what update_freq to change the
        hyperparameter. Can be either "epoch" (default) or "step".
    verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(
        self,
        schedule: Callable,
        hyperparameter: str,
        optimizer: Optional[keras.optimizers.Optimizer] = None,
        update_freq: str = "epoch",
        verbose: int = 0,
    ):
        super(HyperparameterScheduler, self).__init__()
        self.optimizer = optimizer
        self.schedule = schedule
        self.hyperparameter = hyperparameter
        self.verbose = verbose

        if update_freq not in ["epoch", "step"]:
            raise ValueError(
                "HyperparameterScheduler.update_freq can only be 'step' or 'epoch'."
                f" Received value '{update_freq}'"
            )

        self.update_freq = update_freq

    def set_model(self, model: keras.models.Model) -> None:
        super().set_model(model)
        if self.optimizer is None:
            # It is not possible for a model to reach this state and not have
            # an optimizer, so we can safely access it here.
            self.optimizer = model.optimizer

        if not hasattr(self.optimizer, self.hyperparameter):
            raise ValueError(
                f'Optimizer must have a "{self.hyperparameter}" attribute.'
            )

    def set_hyperparameter(self, t: int) -> Any:
        hp = getattr(self.optimizer, self.hyperparameter)
        try:  # new API
            hyperparameter_val = keras.backend.get_value(hp)
            hyperparameter_val = self.schedule(t, hyperparameter_val)
        except TypeError:  # Support for old API for backward compatibility
            hyperparameter_val = self.schedule(t)
        keras.backend.set_value(hp, hyperparameter_val)
        return hp

    def on_batch_begin(
        self, batch: int, logs: Optional[MutableMapping[str, Any]] = None
    ) -> None:
        if self.update_freq == "step":
            # We use optimizer.iterations (i.e. global step), since batch only
            # reflects the batch index in the current epoch.
            hp = self.set_hyperparameter(self.optimizer.iterations.numpy())

            if self.verbose > 0:
                print(
                    f"Batch {self.optimizer.iterations}: {self.hyperparameter} changing"
                    f" to {keras.backend.get_value(hp)}."
                )

    def on_epoch_begin(
        self, epoch: int, logs: Optional[MutableMapping[str, Any]] = None
    ) -> None:
        hp = getattr(self.optimizer, self.hyperparameter)
        try:  # new API
            hyperparameter_val = keras.backend.get_value(hp)
            hyperparameter_val = self.schedule(epoch, hyperparameter_val)
        except TypeError:  # Support for old API for backward compatibility
            hyperparameter_val = self.schedule(epoch)

        keras.backend.set_value(hp, hyperparameter_val)

        if self.verbose > 0:
            print(
                f"Epoch {epoch + 1}: {self.hyperparameter} changing to {keras.backend.get_value(hp)}."
            )

    def on_epoch_end(
        self, epoch: int, logs: Optional[MutableMapping[str, Any]] = None
    ) -> None:
        logs = logs or {}
        hp = getattr(self.optimizer, self.hyperparameter)
        logs[self.hyperparameter] = keras.backend.get_value(hp)
