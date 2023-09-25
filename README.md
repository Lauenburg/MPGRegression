# MPG Regression

This project encapsulates a structured pipeline for the MPG Regression Model. The pipeline is comprised of the following three major components:

1. **CSVDataLoader Class**: Responsible for efficiently loading the CSV data.
2. **LinearRegressionModel Class**: Streamlines the setup for the regression model.
3. **LinearRegressionTrainer Class**: Manages the meticulous training of the model.

**Execution Guide**:

Initiate the pipeline through the `main.py` in the root directory.

**Configuration Guide**:

Configure the model settings via the Hydra `config.yaml` file located in the `conf/` directory.

**SQL Metric Database**:

This repository employs SQLite3 for logging information related to datasets, models, and training data. It organizes data into three tables: Datasets, Models, and Training. The Training table is interlinked with the other two tables through foreign keys, ensuring a structured and relational database setup. Besides configuration information we log for each training run the last training and validation score, the evolution score, and the coefficient of determination (R^2).

To join the Training rows with those of the Datasets and Models tables using foreign keys run:

```sql
SELECT *
FROM Training
JOIN Datasets ON Training.DataSetID = Datasets.DataSetID
JOIN Models ON Training.ModelID = Models.ModelID;
```

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
