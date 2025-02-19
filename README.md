# functional-variant-prediction
A machine learning pipeline using Catboost to predict functional variants.

## 📦 Environment Setup

This project uses **Conda** to manage dependencies. Follow these steps to set up your environment.

###  1. Install Conda
If you don’t have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

###  2. Create the Environment
Run the following command in your terminal to create the Conda environment:

```sh
conda env create -f environment.yml
```
## 🚀 Usage

This repository contains a machine learning pipeline for training and evaluating a **CatBoost Classifier** with **Bayesian hyperparameter optimization**. Follow these steps to run the pipeline:

###  1. Prepare Your Environment
Ensure that you have set up the Conda environment as described in the [Environment Setup](#-environment-setup) section.

Activate the environment:
```sh
conda activate catboost_ml
```
### 2. Prepare Your Features
Make sure you have extracted all features from the [variant annotation pipeline](https://github.com/evotools/nf-VarAnno). The dataset should include a **label** column to indicate the class of each data point.

### 3. Run the Pipeline
The pipeline script supports customization through command-line arguments. An example JSON file defining the hyperparameter search space is provided as **example_params_search_space.json**. This can be adjusted based on your specific use case.  

For example, when training on a dataset with highly imbalanced foreground and background data, you can include a search space for **scale_pos_weight** to help address class imbalance.  

For more details on different hyper-parameters, refer to the [CatBoost documentation](https://catboost.ai/en/docs/references/training-parameters/).


