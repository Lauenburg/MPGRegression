import hydra
import pandas as pd
from omegaconf import DictConfig


class CSVDataLoader:
    def __init__(
        self,
        csv_url: str,
        csv_columns: list,
        csv_na_values: list,
        csv_comment: str,
        csv_sep: str,
        cat2hot_categories: list,
        cat2hot_mapping: list,
        split_frac: float,
        split_random_state: int,
        label_column: str,
    ) -> None:
        self.csv_url = csv_url
        self.csv_columns = csv_columns
        self.csv_na_values = csv_na_values
        self.csv_comment = csv_comment
        self.csv_sep = csv_sep
        self.cat2hot_categories = cat2hot_categories
        self.cat2hot_mapping = cat2hot_mapping
        self.split_frac = split_frac
        self.split_random_state = split_random_state
        self.label_column = label_column

        self.dataset = pd.DataFrame()

        self.train_dataset = pd.DataFrame()
        self.train_features = pd.DataFrame()
        self.train_labels = pd.Series()

        self.test_dataset = pd.DataFrame()
        self.test_features = pd.DataFrame()
        self.test_labels = pd.Series()

    def load(
        self, url: str, columns: list, na_values: list, comment: str, sep: str
    ) -> None:
        """Load the csv dataset from a given URL.

        Args:
            url: URL to the csv dataset
            columns: The columns of the CSV dataset
            na_values: List of value interpreted as NA
            comment: Identifier for comments
            sep: Identifier for separators
        """

        # ensure that the comment variable gets interpreted correctly
        comment = bytes(comment, "utf-8").decode("unicode_escape")

        self.dataset = pd.read_csv(
            url,
            names=columns,
            na_values=na_values,
            comment=comment,
            sep=sep,
            skipinitialspace=True,
        ).dropna()

    def one_hot_encode_columns(self, categories: list, mapping: list) -> None:
        """Convert specified columns to one hot encoding.

        Args:
            categories: The columns that are to be one-hot-encoded
            mapping: New column names in which we split the column for the one-hot-encoding
        """
        for column, mapping in zip(categories, mapping):
            self._one_hot_encode(column, mapping)

    def _one_hot_encode(self, column: str, mapping: list) -> None:
        """The actual encoding of a single column.

        Args:
            categories: The columns that are to be one hot encoded
            mapping: New column names in which we split the column for the one hot encoding
        """
        self.dataset[column] = self.dataset[column].map(
            {i + 1: category for i, category in enumerate(mapping)}
        )
        self.dataset = pd.get_dummies(
            self.dataset, columns=[column], prefix="", prefix_sep="", dtype=int
        )

    def split_data(self, frac: float, random_state: int) -> None:
        """Split the dataset into train and test datasets.

        Args:
            frac: The fraction of the dataset that should be used for training
            random_state: Seed for the random number generator
        """
        self.train_dataset = self.dataset.sample(frac=frac, random_state=random_state)
        self.test_dataset = self.dataset.drop(self.train_dataset.index)

    def split_features_label(self, label_column: str) -> None:
        """Split the label column from the feature columns.

        Args:
            label_column: The column name of the labels
        """
        self.train_features = self.train_dataset.copy()
        self.test_features = self.test_dataset.copy()
        self.train_labels = self.train_features.pop(label_column)
        self.test_labels = self.test_features.pop(label_column)

    def process(self) -> None:
        """Load and process the data."""
        self.load(
            self.csv_url,
            self.csv_columns,
            self.csv_na_values,
            self.csv_comment,
            self.csv_sep,
        )
        self.one_hot_encode_columns(self.cat2hot_categories, self.cat2hot_mapping)
        self.split_data(self.split_frac, self.split_random_state)
        self.split_features_label(self.label_column)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function to run the data pipeline.

    Args:
        cfg: Hydra config object
    """
    pipeline = CSVDataLoader(
        cfg.data.csv_url,
        cfg.data.csv_columns,
        cfg.data.csv_na_values,
        cfg.data.csv_comment,
        cfg.data.csv_sep,
        cfg.data.cat2hot_categories,
        cfg.data.cat2hot_mapping,
        cfg.data.split_frac,
        cfg.data.split_random_state,
        cfg.data.label_column,
    )
    pipeline.process()

    print(pipeline.train_dataset.tail())


if __name__ == "__main__":
    main()
