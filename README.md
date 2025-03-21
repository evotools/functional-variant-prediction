# functional-variant-prediction
A machine learning pipeline using Catboost to predict functional variants.

## 📦 Environment Setup

This project uses **Conda** to manage dependencies. Follow these steps to set up your environment.

###  1. Install Conda
If you don’t have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

###  2. Create the Environment
Run the following command in your terminal to create the Conda environment:

```sh
conda env create -f environment.yaml
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
Make sure you have prepared features for the variants from the [variant annotation pipeline](https://github.com/evotools/nf-VarAnno). The dataset should be tab separated and include a **label** column to indicate the class of each data point.

### 3. Run the Pipeline
The pipeline script supports customization through command-line arguments. An example JSON file defining the hyperparameter search space is provided as **example_params_search_space.json**. This can be adjusted based on your specific use case.  

For more details on different hyper-parameters, refer to the [CatBoost documentation](https://catboost.ai/en/docs/references/training-parameters/).

An example command-line for running the pipeline:
```sh
python machine_learning_pipeline.py --data_path variant_features --param_space example_params_search_space --model_output_path trained_model.cbm --test_chromosomes chr1 
```
**Additional parameters**
**Training and Testing Data**
- `--train_chromosomes` – Specify a **comma-separated list** of chromosomes for training. If not set, the pipeline will use the remaining chromosomes after assigning `--test_chromosomes`.
  
- `--test_chromosomes` – Specify the **chromosomes for testing** (required).
  

**Computation Settings**
- `--task_type` – Choose between **GPU** and **CPU** execution. Default is `GPU`.
  
- `--gpu_ram_part` – Set the fraction of **GPU memory** allocated for CatBoost (default: `0.9`).
  
- `--n_jobs` – Number of **parallel jobs** for BayesSearchCV (default: `1`).
  

**Cross-Validation Settings**
- `--cv_strategy` – Select a **cross-validation strategy**:
  - `"group"` for **GroupKFold** (chromosome-level CV, recommended).
  - `"stratified"` for **StratifiedKFold**.
  
- `--n_groups` – Number of **chromosome-level cross-validation groups** (default: `5`).
 

**Hyperparameter Tuning**
- `--boosting_type` – Specify the boosting type (`Ordered` or `Plain`).
  
- `--n_iter` – Define the **number of iterations** for **BayesSearchCV** (default: `50`).
 
- `--scoring` – Set the **evaluation metric**. Available options:
  - `"accuracy"`, `"precision"`, `"recall"`, `"f1"`, `"roc_auc"`, `"log_loss"`
 
- `--delta` – Set the **minimum acceptable improvement** in model performance (default: `0.0001`).
  

**Tuning and Stopping Criteria**
- `--time_limit` – Set a **time limit (in minutes)** for stopping the tuning process (default: `1440` minutes = 24 hours).
  

**Output Files**
- `--shap_output` – Specify the output file name for the **SHAP summary plot** (default: `"shap_summary_plot.png"`).
  
- `--roc_output` – Specify the output file name for **ROC curves** (default: `"roc_curves.png"`).


It is recommended to run hyper-parameter tuning using **GPUs**, as this significantly improves training efficiency. For reference, we trained our cattle cross-tissues model using the **GPU service** from **Edinburgh international data facility** ([EIDF](https://edinburgh-international-data-facility.ed.ac.uk/services/computing/gpu-service)). With two NVIDIA A100 GPUs, it took approximately 48 hours to complete 50 iterations of hyper-paramter tuning.

### 4. Save and use the trained model
The trained model will be stored at the path specified by `--model_output_path`. If you want to apply the trained model to other variant sets, you can load the model and encode the features of the new variant set based on the encoded training data. We have provdied cattle tissue-specific and cross-tissues models trained on cattle GTEx data. These models can be downloaded here: [https://doi.org/10.5281/zenodo.14901001]. To load the model, simply use the following Python code:
```sh
from catboost import CatBoostClassifier
model = CatBoostClassifier()
model.load_model(trained_model_path)
```
  


