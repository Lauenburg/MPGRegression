# MPG Regression Project

The MPG Regression project offers a structured pipeline for loading a the MPG CSV dataset and training a linear regression model. This document outlines the project's structure, configuration, database logging, prediction service, and setup instructions.

## Project Components

1. **CSVDataLoader Class**: Efficiently loads CSV data.
2. **LinearRegressionModel Class**: Provides a streamlined setup for the regression model.
3. **LinearRegressionTrainer Class**: Handles the detailed training of the model.

## How to Execute

Run `main.py` in the root directory to start the pipeline.

## Configuration

Modify the `config.yaml` file in the `conf/` directory to adjust model settings.

## SQL Metric Database

Utilizes SQLite3 to log dataset, model, and training data, maintaining structure and relations through three tables: Datasets, Models, and Training. Execute the provided SQL command to join the tables.

```sql
SELECT *
FROM Training
JOIN Datasets ON Training.DataSetID = Datasets.DataSetID
JOIN Models ON Training.ModelID = Models.ModelID;
```

## Prediction Service

Retrieve predictions through a REST-API. Replace `<static_key>` and use the provided `curl` command to request a prediction.

```bash
curl -m 70 -X POST https://us-central1-mpgregression.cloudfunctions.net/mpg_regression \
  -H "Content-Type: application/json" \
  -H "api-key: <static_key>" \
  -d '{"instances": [[8.0, 400.0, 230.0, 4278.0, 9.5, 73.0, 2]]}'
```

The instance values should be provided in the following order: `['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']`

> **Note:** Please contact the author to retrieve the static key.

## Setup

Set up the environment of your choice, install the required packages from the `Pipfile`, and execute `main.py`.

#### Setting Up MPGRegression with `pyenv` and `pipenv` on macOS

In the following, we quickly outline how to derive a strong setup using `pyenv` and `pipenv` on macOS.

Why Use pyenv and pipenv?

- **Isolation**: Prevent dependency conflicts with isolated virtual environments.
- **Reproducibility**: Ensure consistent dependency versions with `Pipfile.lock`.
- **Python Version Management**: Easily switch Python versions per project.

1. **Install `pyenv`**:

   ```bash
   brew install pyenv
   ```

2. **Add `pyenv` to your shell**:

   ```bash
   echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
   ```

3. **Install Python version with `pyenv`**:

   ```bash
   pyenv install 3.9.12
   ```

4. **Set global Python version**:

   ```bash
   pyenv global 3.9.12
   ```

5. **Install `pipenv`**:

   ```bash
   pip install pipenv
   ```

6. **Navigate to your project folder**:

   ```bash
   cd /home/user/MPGRegression
   ```

7. **Initialize `pipenv` environment**:

   ```bash
   pipenv --python $(pyenv which python)
   ```

8. **Activate the environment**:

   ```bash
   pipenv shell
   ```

9. **Install the `Pipfile`**:
   ```bash
   pipenv install
   ```
