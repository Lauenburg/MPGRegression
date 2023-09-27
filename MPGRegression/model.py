import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


class LinearRegressionModel(tf.keras.Model):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.normalization_layer = layers.Normalization(axis=-1)
        self.dense = layers.Dense(units=1)

    def call(self, inputs):
        """Specifies the model's call logic.

        Args:
            input: The data to be passed through the network
        """
        x = self.normalization_layer(inputs)
        return self.dense(x)


class LinearRegressionTrainer:
    def __init__(
        self,
        model: LinearRegressionModel,
        learning_rate: float,
        loss_function: str,
        epochs: int,
        validation_split: float,
        train_features: pd.DataFrame,
        train_labels: pd.Series,
        test_features: pd.DataFrame,
        test_labels: pd.Series,
    ) -> None:
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.loss_function = loss_function
        self.epochs = epochs
        self.validation_split = validation_split
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels
        self._adapt_normalization_layer()
        self._compile_model()

    def _compile_model(self):
        """Compile the model."""
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function)

    def _adapt_normalization_layer(self):
        """Adapt the normalization layer to the training features."""
        normalization_layer = self.model.normalization_layer
        normalization_layer.adapt(np.array(self.train_features))

    def train(self) -> pd.DataFrame:
        """Train the model and return the training history."""
        history = self.model.fit(
            self.train_features,
            self.train_labels,
            epochs=self.epochs,
            validation_split=self.validation_split,
            verbose=0,
        )
        return pd.DataFrame(history.history)

    def evaluate(self) -> float:
        """Evaluate the model on test data and return loss."""
        return self.model.evaluate(self.test_features, self.test_labels, verbose=0)
