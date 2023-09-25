from datetime import datetime

import hydra
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig
from sklearn.metrics import r2_score

from MPGRegression.backend import DatabaseManager
from MPGRegression.data import CSVDataLoader
from MPGRegression.model import LinearRegressionModel, LinearRegressionTrainer
from MPGRegression.utils.visualize import plot_loss


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function to run the training pipeline.

    Args:
        cfg: Hydra config object
    """

    # setup the sql registry
    db_manager = DatabaseManager(cfg.database.name, cfg.database.aggregate)

    # setup the tables
    for table in cfg.tables:
        db_manager.execute(table.query)

    # setup the data loader
    dataloader = CSVDataLoader(
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

    # load and process the data
    dataloader.process()

    # write the meta data to the database
    dataset_columns = [
        "DatasetURL",
        "SourceColumns",
        "TargetColumn",
        "DropNA",
        "OneHotEncoding",
        "SplitFraction",
        "SplitRandomState",
    ]
    dataset_values = [
        cfg.data.csv_url,
        str(cfg.data.csv_columns),
        cfg.data.label_column,
        True,
        True,
        cfg.data.split_frac,
        cfg.data.split_random_state,
    ]

    dataset_id = db_manager.insert("Datasets", dataset_columns, dataset_values)

    # setup the model
    model = LinearRegressionModel(
        learning_rate=cfg.model.learning_rate, loss_function=cfg.model.loss_function
    )

    # write the meta data to the database
    models_columns = [
        "ModelType",
        "Normalization",
        "LossFunction",
        "LearningRate",
        "Optimizer",
    ]
    models_values = [
        "LinearRegressionModel",
        True,
        cfg.model.loss_function,
        cfg.model.learning_rate,
        "Adam",
    ]

    model_id = db_manager.insert("Models", models_columns, models_values)

    # track the time need for training - start
    start = datetime.now()
    start = start.strftime("%Y-%m-%d %H:%M:%S")

    # setup the model trainer
    trainer = LinearRegressionTrainer(
        model=model,
        epochs=cfg.train.epochs,
        validation_split=cfg.train.validation_split,
        train_features=dataloader.train_features,
        test_features=dataloader.test_features,
        train_labels=dataloader.train_labels,
        test_labels=dataloader.test_labels,
    )

    history = trainer.train()

    # track the time need for training - end
    stop = datetime.now()
    stop = stop.strftime("%Y-%m-%d %H:%M:%S")

    model.model.save("models")

    if cfg.train.visualize_loss:
        plot_loss(history, cfg.plot.y_lim_bot, cfg.plot.y_lim_top)

    # evaluation
    evaluation_loss = trainer.evaluate()

    print(f"Evaluation loss: {evaluation_loss}")

    # compute r2 score
    predictions = np.transpose(trainer.model.predict(dataloader.test_features))[0]
    targets = dataloader.test_labels.to_numpy()
    r2 = r2_score(targets, predictions)

    print(f"R2 Score: {r2}")

    # write the training and model data to the database
    training_columns = [
        "DataSetID",
        "ModelID",
        "StartTime",
        "EndTime",
        "Epochs",
        "ValidationSplit",
        "FinalTrainLoss",
        "FinalValidationLoss",
        "EvaluationLoss",
        "R2Score",
    ]

    training_values = [
        dataset_id,
        model_id,
        start,
        stop,
        cfg.train.epochs,
        cfg.train.validation_split,
        history["loss"].iloc[-1],
        history["val_loss"].iloc[-1],
        evaluation_loss,
        r2,
    ]

    db_manager.insert("Training", training_columns, training_values)

    # close the database connection
    db_manager.close()


if __name__ == "__main__":
    # run the pipeline
    main()
