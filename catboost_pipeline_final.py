import argparse
import json
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import StratifiedKFold, GroupKFold
from catboost import CatBoostClassifier, Pool
from skopt import BayesSearchCV, space
import logging
import pprint
from skopt.callbacks import DeadlineStopper, DeltaYStopper
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class PrintIterationCallback:
    def __init__(self, scoring='roc_auc'):
        self.iteration = 0
        self.scoring = scoring

    def __call__(self, res):
        self.iteration += 1
        logger.info(f"Iteration {self.iteration}:")
        logger.info(f" - Parameters: {res.x_iters[-1]}")
        logger.info(f" - {self.scoring.upper()}: {res.func_vals[-1]:.4f}")

def encode_features(feature, reference_columns=None):
    """
    Encodes categorical features.
    Args:
        feature: DataFrame containing the features.
        reference_columns: List of columns to ensure consistent encoding. If None, columns will be derived.

    Returns:
        DataFrame with encoded features and column names to use as reference.
    """
    def one_hot_encode(df, col_name, prefix):
        encoded_df = pd.get_dummies(df[col_name], prefix=prefix).astype(int) 
        return encoded_df
    # Flanking sequence encoding
    f_list = feature['flanking_sequence'].apply(lambda x: list(x))
    f_df = pd.DataFrame(x for x in f_list)
    f_df.columns = ['pos1','pos2','pos3','pos4','pos5']
    encode_dic = {"A": "1000", "C": "0100", "G": "0010", "T": "0001"}
    f_df = f_df.replace({"pos1": encode_dic, "pos2": encode_dic, "pos3": encode_dic, "pos4": encode_dic, "pos5": encode_dic}) 
    f_df['flanking_encoded'] = f_df['pos1']+f_df['pos2']+f_df['pos3']+f_df['pos4']+f_df['pos5']
    flanking_encoded_list = f_df['flanking_encoded'].apply(lambda x: list(x))
    flanking_encoded_df = pd.DataFrame(x for x in flanking_encoded_list)
    x=0
    for col in flanking_encoded_df.columns:
        name = "Flanking_sequence_" + str(x)
        flanking_encoded_df = flanking_encoded_df.rename(columns={col:name})
        x = x + 1
    # Allele encoding
    allele_encoded_df = pd.DataFrame({
        'Ancestral_A': (feature['Anc_Der'].str[0] == 'A').astype(int),
        'Ancestral_C': (feature['Anc_Der'].str[0] == 'C').astype(int),
        'Ancestral_G': (feature['Anc_Der'].str[0] == 'G').astype(int),
        'Ancestral_T': (feature['Anc_Der'].str[0] == 'T').astype(int),
        'Derived_A': (feature['Anc_Der'].str[1] == 'A').astype(int),
        'Derived_C': (feature['Anc_Der'].str[1] == 'C').astype(int),
        'Derived_G': (feature['Anc_Der'].str[1] == 'G').astype(int),
        'Derived_T': (feature['Anc_Der'].str[1] == 'T').astype(int)
    })
    # Concatenate encoded features
    feature = pd.concat([feature, flanking_encoded_df, allele_encoded_df], axis=1)
    feature = feature.drop(['flanking_sequence', 'Anc_Der'], axis=1)
    # One-hot encode chromosome and consequence
    chrom_encoded_df = one_hot_encode(feature, 'chrom', 'Chrom')
    consequence_encoded_df = one_hot_encode(feature, 'Consequence', 'Consequence')
    feature = pd.concat([feature, chrom_encoded_df, consequence_encoded_df], axis=1)
    feature = feature.drop(['chrom', 'Consequence', 'variant_id', 'start'], axis=1)
    label = feature[['label']]
    feature = feature.drop('label', axis=1)
    if reference_columns is not None:
        # Add missing columns with zeros and drop extra columns
        for col in reference_columns:
            if col not in feature.columns:
                feature[col] = 0
        feature = feature[reference_columns]
    else:
        reference_columns = feature.columns.tolist()
    return feature, label, reference_columns

# Parse hyperparameter space
def parse_param_space(json_file):
    """
    Parses a JSON file containing hyperparameter definitions and returns a dictionary suitable for skopt.
    Args:
        json_file (str): Path to the JSON file.
    Returns:
        dict: Dictionary containing skopt parameter spaces.
    Raises:
        ValueError: If an unsupported parameter type is encountered.
        FileNotFoundError: If the JSON file is not found.
    """
    try:
        with open(json_file, 'r') as f:
            param_space = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Hyperparameter definition file not found: {json_file}")
    parsed_space = {}
    for param, config in param_space.items():
        if not isinstance(config, dict):
            raise ValueError(f"Invalid configuration format for parameter '{param}': expected dictionary.")
        param_type = config.get('type')
        if param_type == 'Real':
            try:
                low = config['low']
                high = config['high']
                prior = config.get('prior', 'uniform')
                parsed_space[param] = space.Real(low, high, prior=prior)
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid configuration for Real parameter '{param}': {e}")
        elif param_type == 'Integer':
            try:
                low = config['low']
                high = config['high']
                parsed_space[param] = space.Integer(low, high)
            except (KeyError, TypeError) as e:
                raise ValueError(f"Invalid configuration for Integer parameter '{param}': {e}")
        elif param_type == 'Categorical':
            try:
                categories = config['categories']
                parsed_space[param] = space.Categorical(categories)
            except KeyError as e:
                raise ValueError(f"Invalid configuration for Categorical parameter '{param}': missing 'categories' key.")
        else:
            raise ValueError(f"Unsupported parameter type: {param_type}")
    return parsed_space

def report_perf(optimizer, X, y, title="model", callbacks=None, groups=None):
    """
    Report the performance of the hyper-parameter tuning process.

    Args:
        optimizer: The Bayesian optimization object used for hyperparameter tuning.
        X: Feature matrix for model training.
        y: Target labels corresponding to the feature matrix.
        title: Title of the model, used in logging output. Defaults to "model".
        callbacks: List of callback functions to be used during optimization. If None, only `PrintIterationCallback` will be used. Defaults to None.
        groups: Group labels for cross-validation splits, required when using `GroupKFold` or similar group-based CV methods. Defaults to None.
    Return:
        dict: The best hyperparameters found during tuning.
    """
    start = time()
    iteration_logger = PrintIterationCallback(scoring=args.scoring)

    if callbacks is not None:
        callbacks = callbacks + [iteration_logger]
    else:
        callbacks = [iteration_logger]

    optimizer.fit(X, y, groups=groups, callback=callbacks)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    logger.info(
    ("%s took %.2f seconds, candidates checked: %d, best CV score: %.3f "
     + u"\u00B1" + " %.3f") % (
        title,
        time() - start,
        len(optimizer.cv_results_['params']),
        best_score,
        best_score_std
    )
    )
    logger.info('Best parameters:')
    logger.info(pprint.pformat(best_params))
    logger.info("")

    return best_params

def generate_shap_plot(trained_model, X_train, output_file="shap_summary_plot.png"):
    """
    Generate and save the SHAP summary plot.

    Args:
        trained_model: The final trained CatBoost model.
        X_train: Feature matrix for the training dataset.
        output_file: Path to save the generated SHAP plot. Defaults to "shap_summary_plot.png".
    Returns:
        None
    """
    logger.info("Generating SHAP summary plot...")
    explainer = shap.Explainer(trained_model)
    shap_values = explainer(X_train)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_train, 
        show=False, 
        plot_type="dot"
    )
    plt.title("SHAP Summary Plot for Feature Importance", fontsize=16, fontweight="bold")
    plt.xlabel("SHAP Value (Impact on Model Output)", fontsize=14)
    plt.ylabel("Features", fontsize=14)
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=400)
    logger.info(f"SHAP summary plot saved as '{output_file}'.")

def generate_roc_curves(trained_model, X_train, y_train, X_test, y_test, best_params, train_chromosomes=None, train_data=None, cv_strategy="group", output_file="roc_curves.png"):
    """
    Generates and saves ROC (Receiver Operating Characteristic) curves for both cross-validation and test set performance evaluation.

    Args:
        trained_model: The final trained CatBoost model used for evaluation.        
        X_train: Feature matrix for the training dataset.        
        y_train: Labels for the training dataset.       
        X_test: Feature matrix for the test dataset.        
        y_test: Labels for the test dataset.        
        best_params (dict): Dictionary containing the best hyperparameters obtained from tuning.        
        train_chromosomes: Set of chromosome identifiers used for training, required if `cv_method="group"`. Defaults to None.        
        train_data: Original training dataset containing chromosome information, used for chromosome-level cross-validation. Defaults to None.        
        cv_method: Cross-validation strategy to use:
            - "group": Uses `GroupKFold` with chromosomes as groups.
            - Other values: Uses stratified k-fold cross-validation. Defaults to "group".        
        output_file: Path to save the generated ROC curve plot. Defaults to "roc_curves.png".
    Returns:
        None
    """
    logger.info("Generating ROC curves...")
    plt.figure(figsize=(10, 8))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    # Choose cross-validation method
    if cv_strategy == "group":
        # Chromosome-level cross-validation
        cv = GroupKFold(n_splits=max(2, min(len(train_chromosomes), 5)))
        groups = train_data['chrom']  # Use chromosome as groups
        cv_splits = cv.split(X_train, y_train, groups=groups)
        cv_type_label = "Chromosome-Level CV"
    else:
        # Stratified k-fold cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = cv.split(X_train, y_train)
        cv_type_label = "K-Fold CV"

    # Plot individual cross-validation ROC curves in grey
    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        train_X, val_X = X_train.iloc[train_idx], X_train.iloc[val_idx]
        train_y, val_y = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model = CatBoostClassifier(**best_params, task_type=args.task_type, verbose=0, random_state=42)
        model.fit(train_X, train_y)
        val_proba = model.predict_proba(val_X)[:, 1]
        fpr, tpr, _ = roc_curve(val_y, val_proba)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        logger.info(f"Fold {fold} AUC: {roc_auc:.4f}")
        label = "Cross-validation ROC" if fold == 1 else None
        plt.plot(fpr, tpr, color="grey", alpha=0.3, lw=1, label=label)
    plt.legend()

    # Plot mean ROC curve in orange with a solid line
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color="orange", lw=2, linestyle="-", label=f"Mean ROC ({cv_type_label}, AUC = {mean_auc:.2f})")
    logger.info(f"Mean CV AUC: {mean_auc:.4f}")

    # Plot final test ROC curve in teal with a solid line
    predictions_proba = trained_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predictions_proba)
    test_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color="teal", lw=2, linestyle="-", label=f"Test ROC (AUC = {test_auc:.2f})")
    logger.info(f"Test AUC: {test_auc:.4f}")

    # Plot the diagonal line
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2)

    # Customize the plot
    plt.grid(alpha=0.4)
    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)
    plt.title(f"ROC Curves ({cv_type_label})", fontsize=18, fontweight="bold")
    plt.legend(loc="lower right", fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=400)
    logger.info(f"ROC curves saved as '{output_file}'.")

def main(args):
    logger.info("Loading dataset...")
    data = pd.read_table(args.data_path, sep='\t')
    logger.info(f"Dataset loaded with shape: {data.shape}")

    logger.info("Splitting data by chromosome...")
    test_chromosomes = set(args.test_chromosomes.split(","))
    all_chromosomes = set(data['chrom'].unique())
    if args.train_chromosomes:
        train_chromosomes = set(args.train_chromosomes.split(","))
    else:
        train_chromosomes = all_chromosomes - test_chromosomes
        logger.info("train_chromosomes not specified. Automatically set to:")
        logger.info(f"{train_chromosomes}")

    train_data = data[data["chrom"].isin(train_chromosomes)].reset_index(drop=True)
    test_data = data[data["chrom"].isin(test_chromosomes)].reset_index(drop=True)
    logger.info(f"Training set size: {train_data.shape}")
    logger.info(f"Test set size: {test_data.shape}")

    logger.info("Encoding training set...")
    X_train, y_train, reference_columns = encode_features(train_data)
    logger.info("Encoding test set...")
    X_test, y_test, _ = encode_features(test_data, reference_columns)

    logger.info(f"Encoded training set shape: {X_train.shape}")
    logger.info(f"Encoded test set shape: {X_test.shape}")

    # Determine cross-validation strategy
    if args.cv_strategy == "group":
        cv = GroupKFold(n_splits=min(len(train_chromosomes), args.n_groups))
        groups = train_data['chrom']  # Use chromosome as groups
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        groups = None

    param_space = parse_param_space(args.param_space)

    catboost_model = CatBoostClassifier(
        task_type=args.task_type,
        verbose=0,
        random_state=42,
        gpu_ram_part=args.gpu_ram_part,
        boosting_type=args.boosting_type
    )

    # Create CatBoost training and validation pools
    train_pool = Pool(X_train, y_train)

    # Hyperparameter tuning with BayesSearchCV
    bayes_search = BayesSearchCV(
        estimator=catboost_model,
        search_spaces=param_space,
        cv=cv,
        n_iter=args.n_iter,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        random_state=42
    )
    overdone_control = DeltaYStopper(delta=args.delta)
    time_limit_control = DeadlineStopper(total_time=60 * args.time_limit)
    logger.info("Starting hyperparameter tuning...")
    best_params = report_perf(bayes_search, train_pool.get_features(), train_pool.get_label(), 
                              title="CatBoost Hyperparameter Tuning",
                              callbacks=[overdone_control, time_limit_control], groups=groups)

    # Train the final model with the best hyper-parameters and test on the leave-out chromosomes
    logger.info("Training the final model...")
    final_model = CatBoostClassifier(
        **best_params,
        task_type=args.task_type,
        verbose=0,
        random_state=42,
        gpu_ram_part=args.gpu_ram_part,
        boosting_type=args.boosting_type
    )
    final_model.fit(X_train, y_train)
    final_model.save_model(args.model_output_path)
    logger.info(f"Model saved to {args.model_output_path}")

    # Evaluate the final model on the test set
    logger.info("Evaluating the final model...")
    test_score = final_model.eval_metrics(Pool(X_test, y_test), metrics=['Accuracy', 'AUC', 'F1'])
    logger.info(f"Test set Accuracy: {test_score['Accuracy'][-1]:.4f}")
    logger.info(f"Test set AUC: {test_score['AUC'][-1]:.4f}")
    logger.info(f"Test set F1: {test_score['F1'][-1]:.4f}")

    # Generate SHAP summary plot
    generate_shap_plot(final_model, X_train, output_file=args.shap_output)

    # Plot ROC curve for cross-validation and test set
    generate_roc_curves(
        final_model, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        best_params,
        train_chromosomes=train_chromosomes,
        train_data=train_data, 
        cv_strategy=args.cv_strategy, 
        output_file=args.roc_output
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize a CatBoost model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input dataset CSV.")
    parser.add_argument("--param_space", type=str, required=True, help="Path to the JSON file with hyperparameter space.")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path to save the final trained model.")
    parser.add_argument("--train_chromosomes", type=str, default=None, help="Comma-separated list of chromosomes for the training set. If not specified, uses the remaining chromosomes.")
    parser.add_argument("--test_chromosomes", type=str, required=True, help="Comma-separated list of chromosomes for the test set.")
    parser.add_argument("--task_type", type=str, default="GPU", choices=["GPU", "CPU"], help="CatBoost task type.")
    parser.add_argument("--cv_strategy", type=str, default="group", choices=["stratified", "group"], help="Cross-validation strategy: 'stratified' for StratifiedKFold, 'group' for GroupKFold.")
    parser.add_argument("--gpu_ram_part", type=float, default=0.9, help="Fraction of GPU memory allocated for CatBoost.")
    parser.add_argument("--boosting_type", type=str, default="Ordered", help="Boosting type in Catboost")   
    parser.add_argument("--n_iter", type=int, default=50, help="Number of iterations for BayesSearchCV.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs that can run in parallel in BayesSearchCV")
    parser.add_argument("--n_groups", type=int, default=5, help="Number of groups for performing chromsome-level cross-validation")
    parser.add_argument("--scoring", type=str, default="roc_auc", choices=["accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"], help="Scoring metric for model evaluation.")
    parser.add_argument("--delta", type=float, default=0.0001, help="Minimal acceptable improvement in model performance.")
    parser.add_argument("--time_limit", type=int, default=1440, help="Time limit for stopping the tuning process, specified in minutes.")
    parser.add_argument("--shap_output", type=str, default="shap_summary_plot.png", help="Output file for SHAP plot.")
    parser.add_argument("--roc_output", type=str, default="roc_curves.png", help="Output file for ROC curves.")

    args = parser.parse_args()
    main(args)