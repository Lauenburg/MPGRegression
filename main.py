import hydra
from omegaconf import DictConfig

from MPGRegression.data import CSVDataLoader
from MPGRegression.model import LinearRegressionModel, LinearRegressionTrainer
from MPGRegression.utils.visualize import plot_loss


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function to run the training pipeline.

    Args:
        cfg: Hydra config object
    """

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

    # setup the model
    model = LinearRegressionModel(
        learning_rate=cfg.model.learning_rate, loss_function=cfg.model.loss_function
    )

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

    model.model.save("models")

    if cfg.train.visualize_loss:
        plot_loss(history, cfg.plot.y_lim_bot, cfg.plot.y_lim_top)

    # evaluation
    evaluation_loss = trainer.evaluate()

    print(f"Evaluation loss: {evaluation_loss}")


if __name__ == "__main__":
    # run the pipeline
    main()
