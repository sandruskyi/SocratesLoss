"""
@User: sandruskyi
"""
import sys
"""Used to replace the EarlyStopping feature from Tensorflow in PyTorch"""

__all__ = ['MetricMonitor']

class MetricMonitor:
    """Used to replace the EarlyStopping feature from Tensorflow in PyTorch"""
    def __init__(self, patience, min_delta=0.0, mode="min"):
        self.patience = patience
        self.mode = mode
        if mode != "min" and mode != "max":
            raise ValueError(f"Parameter 'mode' needs to be 'min' or 'max' not '{mode}'")
        self.min_delta = min_delta
        self.best_metric = sys.float_info.min if self.mode == "max" else sys.float_info.max
        self.bad_epochs = 0
        self.best_weights = None

    def __call__(self, metric_value, weights, *args, **kwargs):
        """When called checks the metric value and save the weights if better. Returns a boolean indicating whether to
        trigger early stopping or not"""
        if self.mode == "min":
            if metric_value + self.min_delta < self.best_metric:
                self.best_metric = metric_value
                self.bad_epochs = 0
                self.best_weights = weights
            else:
                self.bad_epochs += 1
        elif self.mode == "max":
            if metric_value - self.min_delta > self.best_metric:
                self.best_metric = metric_value
                self.bad_epochs = 0
                self.best_weights = weights
            else:
                self.bad_epochs += 1
        return self.bad_epochs > self.patience

    def get_best_weights(self):
        """Returns the best model parameters thus far. Often called after early stopping is
        triggered to restore weights"""
        return self.best_weights


if __name__ == "__main__":
    pass