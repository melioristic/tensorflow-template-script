import tensorflow as tf

class NameOfMetric(tf.keras.metrics.Metric):
    def __init__(self, name="name_of_metric", **kwargs) -> None:
        super(NameOfMetric, self).__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred):
        
        # Compute the metric here

        self.NOM = 1 # something 

    def result(self):
        return self.NOM
