import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers


class LinearRegressionModel:
    def __init__(self, learning_rate: float, loss_function: str) -> None:
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build and compile the Linear Regression model."""
        normalization_layer = layers.Normalization(axis=-1)
        model = tf.keras.Sequential([normalization_layer, layers.Dense(units=1)])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=self.loss_function)
        return model


class LinearRegressionTrainer:
    def __init__(
        self,
        model: LinearRegressionModel,
        epochs: int,
        validation_split: float,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        train_labels: pd.Series,
        test_labels: pd.Series,
    ) -> None:
        self.model = model.model
        self.epochs = epochs
        self.validation_split = validation_split
        self.train_features = train_features
        self.test_features = test_features
        self.train_labels = train_labels
        self.test_labels = test_labels
        self._adapt_normalization_layer()

    def _adapt_normalization_layer(self):
        """Adapt the normalization layer to the training features."""
        normalization_layer = self.model.layers[0]
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
