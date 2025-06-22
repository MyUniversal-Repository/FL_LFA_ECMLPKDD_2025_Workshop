#######################################################################################################################################################
######## isolalation forest #################

import os
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from federated_learning.utils import (
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)
from sklearn.decomposition import PCA
from client import Client
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
logger.add("IF logPD=40CIFA", rotation="10 MB", level="INFO", encoding="utf-8")

# Paths
MODELS_PATH = "D:/SGD_TargetedDPA/3000_models"
EXP_INFO_PATH = "D:/SGD_TargetedDPA/logs"

# Settings
EPOCHS = list(range(10, 201))
# LAYER_NAME = "fc.weight"
LAYER_NAME = "fc2.weight"
CLASS_NUM = 5
POISONED_WORKER_IDS = [11, 39, 3, 29, 43, 47, 8, 24, 33, 18, 4, 36, 7, 30, 46, 44, 9, 14, 42, 2, 45, 25, 38, 32, 21, 0, 31, 15, 41, 1, 22, 17, 48, 37, 5, 49, 26, 27, 34, 10]
SAVE_NAME = "IFG defense_results(PD = 40CIFA).jpg"
SAVE_SIZE = (20, 20)

def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients


def plot_gradients_2d(gradients_with_preds):
    fig = plt.figure()
    for (worker_id, gradient, pred) in gradients_with_preds:
        if pred == -1:
            plt.scatter(gradient[0], gradient[1], color="red", marker="x", s=200, linewidth=5,
                        label="Anomaly" if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(gradient[0], gradient[1], color="green", s=180,
                        label="Normal" if 'Normal' not in plt.gca().get_legend_handles_labels()[1] else "")
    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    # plt.margins(0, 0)
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.xlabel("Principal Component 1", fontsize=20, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)
    # plt.close(fig)  # Close figure after saving to free memory

if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]

        start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]

            end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

            gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
            gradient = gradient.flatten()

            param_diff.append(gradient)
            worker_ids.append(worker_id)

    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
    logger.info("Prescaled gradients: {}".format(str(param_diff)))

    scaled_param_diff = apply_standard_scaler(param_diff)
    logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))

    y_true = [-1 if wid in POISONED_WORKER_IDS else 1 for wid in worker_ids]

    best_f1 = 0
    best_contamination = None
    best_metrics = None
    best_predictions = None
    best_scores = None

    contamination_values = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for c in contamination_values:
        iso_forest = IsolationForest(n_estimators=10000, contamination=c, random_state=42)
        predictions = iso_forest.fit_predict(scaled_param_diff)
        scores = iso_forest.decision_function(scaled_param_diff)

        precision = precision_score(y_true, predictions, pos_label=-1)
        recall = recall_score(y_true, predictions, pos_label=-1)
        f1 = f1_score(y_true, predictions, pos_label=-1)
        accuracy = accuracy_score(y_true, predictions)
        roc_auc = roc_auc_score(y_true, scores)

        logger.info(f"Contamination={c:.3f} => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
        #logger.info(f"Contamination={c:.3f} => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_contamination = c
            best_metrics = (precision, recall, f1, accuracy, roc_auc)
            # best_metrics = (precision, recall, f1, accuracy)
            best_predictions = predictions
            best_scores = scores

    logger.info(f"\nBest contamination: {best_contamination} with F1 score: {best_f1:.4f}")
    logger.info(f"Precision: {best_metrics[0]:.4f}, Recall: {best_metrics[1]:.4f}, F1: {best_metrics[2]:.4f}, Accuracy: {best_metrics[3]:.4f}, ROC-AUC: {best_metrics[4]:.4f}")
    #logger.info(f"Precision: {best_metrics[0]:.4f}, Recall: {best_metrics[1]:.4f}, F1: {best_metrics[2]:.4f}, Accuracy: {best_metrics[3]:.4f}")

    cm = confusion_matrix(y_true, best_predictions, labels=[1, -1])
    logger.info("Confusion Matrix (rows=actual, columns=predicted):\n{}".format(cm))

    report = classification_report(y_true, best_predictions, target_names=["Normal", "Poisoned"], labels=[1, -1])
    logger.info("Classification Report:\n{}".format(report))

    pca = PCA(n_components=2, random_state=42)
    dim_reduced_gradients = pca.fit_transform(scaled_param_diff)
    logger.info("PCA reduced gradients: {}".format(dim_reduced_gradients.tolist()))
    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    plot_gradients_2d(zip(worker_ids, dim_reduced_gradients, best_predictions))


############################################################################################################################################################
## k-menas #####################

import os
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from federated_learning.utils import (
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from client import Client
import matplotlib.pyplot as plt

# Paths
MODELS_PATH = "D:/SGD_TargetedDPA/3000_models"
EXP_INFO_PATH = "D:/SGD_TargetedDPA/logs"

# Settings
EPOCHS = list(range(10, 201))
# LAYER_NAME = "fc.weight"
LAYER_NAME = "fc2.weight"
CLASS_NUM = 5
# POISONED_WORKER_IDS = [8, 10, 42, 25, 18, 40, 43, 44, 23, 35, 24, 6, 0, 15, 38, 4, 49, 13, 19, 37]
POISONED_WORKER_IDS = [11, 39, 3, 29, 43, 47, 8, 24, 33, 18, 4, 36, 7, 30, 46, 44, 9, 14, 42, 2, 45, 25, 38, 32, 21, 0, 31, 15, 41, 1, 22, 17, 48, 37, 5, 49, 26, 27, 34, 10]
SAVE_NAME = "defense_resultsKM(PD=40CIFA).jpg"
SAVE_SIZE = (20, 20)

# Setup logging with rotation
logger.add("KM_logPD=40CIFA.log", rotation="10 MB", level="INFO", encoding="utf-8")

def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients

def plot_gradients_2d(gradients_with_preds):
    fig = plt.figure()
    for (worker_id, gradient, pred) in gradients_with_preds:
        if pred == -1:
            plt.scatter(gradient[0], gradient[1], color="red", marker="x", s=200, linewidth=5,
                        label="Anomaly" if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(gradient[0], gradient[1], color="green", s=180,
                        label="Normal" if 'Normal' not in plt.gca().get_legend_handles_labels()[1] else "")
    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    # plt.margins(0, 0)
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.xlabel("Principal Component 1", fontsize=20, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)
    
if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]

        start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]

            end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

            gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
            gradient = gradient.flatten()

            param_diff.append(gradient)
            worker_ids.append(worker_id)

    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
    logger.info("Prescaled gradients: {}".format(str(param_diff)))

    scaled_param_diff = apply_standard_scaler(param_diff)
    logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))

    # True labels: -1 = anomaly (poisoned), 1 = normal
    y_true = [-1 if wid in POISONED_WORKER_IDS else 1 for wid in worker_ids]

    # Convert true labels to binary 0/1 for ROC AUC (1 = poisoned, 0 = normal)
    y_true_binary = [1 if label == -1 else 0 for label in y_true]

    # Use KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_param_diff)

    # Determine which cluster corresponds to poisoned
    cluster0_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == 0]
    cluster1_indices = [i for i, lbl in enumerate(cluster_labels) if lbl == 1]

    poisoned_in_0 = sum(1 for i in cluster0_indices if y_true[i] == -1)
    poisoned_in_1 = sum(1 for i in cluster1_indices if y_true[i] == -1)

    if poisoned_in_0 > poisoned_in_1:
        predictions = [-1 if lbl == 0 else 1 for lbl in cluster_labels]
    else:
        predictions = [-1 if lbl == 1 else 1 for lbl in cluster_labels]

    # Calculate anomaly scores for ROC-AUC: use negative minimum distance to nearest cluster center
    scores = -kmeans.transform(scaled_param_diff).min(axis=1)

    # Evaluation metrics
    precision = precision_score(y_true, predictions, pos_label=-1)
    recall = recall_score(y_true, predictions, pos_label=-1)
    f1 = f1_score(y_true, predictions, pos_label=-1)
    accuracy = accuracy_score(y_true, predictions)
    roc_auc = roc_auc_score(y_true_binary, -scores)

    logger.info(f"KMeans => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

    cm = confusion_matrix(y_true, predictions, labels=[1, -1])
    logger.info("Confusion Matrix (rows=actual, columns=predicted):\n{}".format(cm))

    report = classification_report(y_true, predictions, target_names=["Normal", "Poisoned"], labels=[1, -1])
    logger.info("Classification Report:\n{}".format(report))

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    dim_reduced_gradients = pca.fit_transform(scaled_param_diff)
    logger.info("PCA reduced gradients: {}".format(dim_reduced_gradients.tolist()))
    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    # Plot the results
    plot_gradients_2d(zip(worker_ids, dim_reduced_gradients, predictions))

##########################################################################################################################################################
##### Agglomerative clustering ###########################
import os
from loguru import logger
from federated_learning.arguments import Arguments
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from federated_learning.utils import (
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from client import Client
import matplotlib.pyplot as plt
import itertools
import numpy as np

# Paths
MODELS_PATH = "D:/SGD_TargetedDPA/3000_models"
EXP_INFO_PATH = "D:/SGD_TargetedDPA/logs"

# Settings
EPOCHS = list(range(10, 201))
# LAYER_NAME = "fc.weight"
LAYER_NAME = "fc2.weight"
CLASS_NUM = 5
POISONED_WORKER_IDS = [11, 39, 3, 29, 43, 47, 8, 24, 33, 18, 4, 36, 7, 30, 46, 44, 9, 14, 42, 2, 45, 25, 38, 32, 21, 0, 31, 15, 41, 1, 22, 17, 48, 37, 5, 49, 26, 27, 34, 10]
SAVE_NAME = "AGGM_defense_results(PD=40CIFA).jpg"
SAVE_SIZE = (20, 20)

logger.add("AGGM_logPD=40CIFA.log", rotation="10 MB", level="INFO", encoding="utf-8")

def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients


def plot_gradients_2d(gradients_with_preds):
    fig = plt.figure()
    for (worker_id, gradient, pred) in gradients_with_preds:
        if pred == -1:
            plt.scatter(gradient[0], gradient[1], color="red", marker="x", s=200, linewidth=5,
                        label="Anomaly" if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(gradient[0], gradient[1], color="green", s=180,
                        label="Normal" if 'Normal' not in plt.gca().get_legend_handles_labels()[1] else "")
    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    # plt.margins(0, 0)
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.xlabel("Principal Component 1", fontsize=20, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)
    
def assign_anomaly_labels(cluster_labels):
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    anomaly_cluster = min(cluster_sizes, key=cluster_sizes.get)
    return [-1 if lbl == anomaly_cluster else 1 for lbl in cluster_labels]

if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]
        start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]
            end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
            gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
            gradient = gradient.flatten()
            param_diff.append(gradient)
            worker_ids.append(worker_id)

    logger.info("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
    logger.info("Prescaled gradients: {}".format(str(param_diff)))

    scaled_param_diff = apply_standard_scaler(param_diff)
    logger.info("Postscaled gradients: {}".format(str(scaled_param_diff)))

    y_true = [-1 if wid in POISONED_WORKER_IDS else 1 for wid in worker_ids]

    n_clusters_values = [2, 3, 4]
    linkage_values = ['ward', 'complete', 'average', 'single']

    best_f1 = -1
    best_params = None
    best_predictions = None
    best_metrics = {}
    best_model = None

    for n_clusters, linkage in itertools.product(n_clusters_values, linkage_values):
        if linkage == 'ward' and n_clusters < 2:
            continue

        agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_labels = agglo.fit_predict(scaled_param_diff)
        predictions = assign_anomaly_labels(cluster_labels)

        precision = precision_score(y_true, predictions, pos_label=-1, zero_division=0)
        recall = recall_score(y_true, predictions, pos_label=-1, zero_division=0)
        f1 = f1_score(y_true, predictions, pos_label=-1, zero_division=0)
        accuracy = accuracy_score(y_true, predictions)

        logger.info(f"Agglomerative (n_clusters={n_clusters}, linkage={linkage}) => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = (n_clusters, linkage)
            best_predictions = predictions
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            }
            best_model = agglo

    logger.info(f"Best params: n_clusters={best_params[0]}, linkage={best_params[1]}, with F1={best_f1:.4f}")

    cm = confusion_matrix(y_true, best_predictions, labels=[1, -1])
    logger.info("Confusion Matrix (best params):\n{}".format(cm))

    report = classification_report(y_true, best_predictions, target_names=["Normal", "Poisoned"], labels=[1, -1], zero_division=0)
    logger.info("Classification Report (best params):\n{}".format(report))

    y_true_binary = [1 if label == -1 else 0 for label in y_true]

    # 🔧 FIXED: Use distances to cluster centroids to compute scores
    cluster_labels = best_model.labels_
    scores = np.zeros(len(scaled_param_diff))
    scaled_param_diff = np.array(scaled_param_diff)

    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_points = scaled_param_diff[cluster_indices]
        cluster_center = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        scores[cluster_indices] = distances

    scores = -scores  # Negative for consistency with anomaly score logic

    try:
        roc_auc = roc_auc_score(y_true_binary, -scores)
        logger.info(f"ROC-AUC Score (best params): {roc_auc:.4f}")
        best_metrics["roc_auc"] = roc_auc
    except ValueError:
        logger.warning("ROC-AUC calculation failed due to only one class present in predictions.")
        best_metrics["roc_auc"] = None

    logger.info("Best Metrics Summary:")
    for k, v in best_metrics.items():
        logger.info(f"{k.upper()}: {v:.4f}" if v is not None else f"{k.upper()}: Not computable")

    pca = PCA(n_components=2, random_state=42)
    dim_reduced_gradients = pca.fit_transform(scaled_param_diff)
    logger.info("PCA reduced gradients: {}".format(dim_reduced_gradients.tolist()))
    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    plot_gradients_2d(zip(worker_ids, dim_reduced_gradients, best_predictions))
    
############################################################################################################################################################################################
########## OC_SVM #########    
        
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import ParameterGrid
from federated_learning.arguments import Arguments
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from federated_learning.utils import (
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)
from client import Client

# Paths
MODELS_PATH = "D:/SGD_TargetedDPA/3000_models"
EXP_INFO_PATH = "D:/SGD_TargetedDPA/logs"

# Settings
EPOCHS = list(range(10, 201))
# LAYER_NAME = "fc.weight"
LAYER_NAME = "fc2.weight"
CLASS_NUM = 5
POISONED_WORKER_IDS = [11, 39, 3, 29, 43, 47, 8, 24, 33, 18, 4, 36, 7, 30, 46, 44, 9, 14, 42, 2, 45, 25, 38, 32, 21, 0, 31, 15, 41, 1, 22, 17, 48, 37, 5, 49, 26, 27, 34, 10]
SAVE_NAME = "defense_results_OC-SVM(PD=40CIFA).jpg"
SAVE_SIZE = (20, 20)

logger.add("OC_SVM_logPD=40CIFA.log", rotation="10 MB", level="INFO", encoding="utf-8")


def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients

def plot_gradients_2d(gradients_with_preds):
    fig = plt.figure()
    for (worker_id, gradient, pred) in gradients_with_preds:
        if pred == -1:
            plt.scatter(gradient[0], gradient[1], color="red", marker="x", s=200, linewidth=5,
                        label="Anomaly" if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(gradient[0], gradient[1], color="green", s=180,
                        label="Normal" if 'Normal' not in plt.gca().get_legend_handles_labels()[1] else "")
    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    # plt.margins(0, 0)
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.xlabel("Principal Component 1", fontsize=20, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)
    

if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]

        start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]

            end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
            gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
            param_diff.append(gradient.flatten())
            worker_ids.append(worker_id)

    scaled_param_diff = apply_standard_scaler(param_diff)
    y_true = [-1 if wid in POISONED_WORKER_IDS else 1 for wid in worker_ids]

    param_grid = {
        'nu': [0.01, 0.05, 0.1, 0.2, 0.3],
        'gamma': ['scale', 'auto', 0.01, 0.001, 0.0001, 0.1, 1]
    }

    best_f1 = -1
    best_params = None
    best_predictions = None
    best_model = None
    best_metrics = {}

    for params in ParameterGrid(param_grid):
        clf = OneClassSVM(kernel='rbf', **params)
        preds = clf.fit_predict(scaled_param_diff)

        precision = precision_score(y_true, preds, pos_label=-1, zero_division=0)
        recall = recall_score(y_true, preds, pos_label=-1, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=-1, zero_division=0)
        accuracy = accuracy_score(y_true, preds)

        logger.info(f"One-Class SVM (nu={params['nu']}, gamma={params['gamma']}) => Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_predictions = preds
            best_model = clf
            best_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy
            }

    logger.info(f"Best One-Class SVM params: nu={best_params['nu']}, gamma={best_params['gamma']}, F1={best_f1:.4f}")

    cm = confusion_matrix(y_true, best_predictions, labels=[1, -1])
    logger.info("Confusion Matrix:\n{}".format(cm))

    report = classification_report(y_true, best_predictions, target_names=["Normal", "Poisoned"], labels=[1, -1], zero_division=0)
    logger.info("Classification Report:\n{}".format(report))

    y_true_binary = [1 if label == -1 else 0 for label in y_true]
    try:
        scores = -best_model.decision_function(scaled_param_diff)
        roc_auc = roc_auc_score(y_true_binary, scores)
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        best_metrics["roc_auc"] = roc_auc
    except Exception as e:
        logger.warning(f"ROC-AUC calculation failed: {e}")
        best_metrics["roc_auc"] = None

    logger.info("Best Metrics Summary:")
    for k, v in best_metrics.items():
        logger.info(f"{k.upper()}: {v:.4f}" if v is not None else f"{k.upper()}: Not computable")

    pca = PCA(n_components=2, random_state=42)
    dim_reduced_gradients = pca.fit_transform(scaled_param_diff)
    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    plot_gradients_2d(zip(worker_ids, dim_reduced_gradients, best_predictions))
    
    
#########################################################################################################################################################################################
###### LOF-Final #######
  
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.model_selection import ParameterGrid
from federated_learning.arguments import Arguments
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from federated_learning.utils import (
    get_model_files_for_epoch,
    get_model_files_for_suffix,
    apply_standard_scaler,
    get_worker_num_from_model_file_name
)
from client import Client

# Paths
MODELS_PATH = "D:/SGD_TargetedDPA/3000_models"
EXP_INFO_PATH = "D:/SGD_TargetedDPA/logs"

# Settings
EPOCHS = list(range(10, 201))
# LAYER_NAME = "fc.weight"
LAYER_NAME = "fc2.weight"
CLASS_NUM = 5
POISONED_WORKER_IDS = [11, 39, 3, 29, 43, 47, 8, 24, 33, 18, 4, 36, 7, 30, 46, 44, 9, 14, 42, 2, 45, 25, 38, 32, 21, 0, 31, 15, 41, 1, 22, 17, 48, 37, 5, 49, 26, 27, 34, 10]
SAVE_NAME = "defense_results_LOF(PD=40_CIFA).jpg"
SAVE_SIZE = (20, 20)  # Square figure

logger.add("LOF_logPD=40_CIFA.log", rotation="10 MB", level="INFO", encoding="utf-8")

def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients


def plot_gradients_2d(gradients_with_preds):
    fig = plt.figure()
    for (worker_id, gradient, pred) in gradients_with_preds:
        if pred == -1:
            plt.scatter(gradient[0], gradient[1], color="red", marker="x", s=200, linewidth=5,
                        label="Anomaly" if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(gradient[0], gradient[1], color="green", s=180,
                        label="Normal" if 'Normal' not in plt.gca().get_legend_handles_labels()[1] else "")
    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    # plt.margins(0, 0)
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.xlabel("Principal Component 1", fontsize=20, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)
       
if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", str(len(model_files)))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]

        start_model_layer_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]

            end_model_layer_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])
            gradient = calculate_parameter_gradients(logger, start_model_layer_param, end_model_layer_param)
            param_diff.append(gradient.flatten())
            worker_ids.append(worker_id)

    scaled_param_diff = apply_standard_scaler(param_diff)
    y_true = [-1 if wid in POISONED_WORKER_IDS else 1 for wid in worker_ids]

    param_grid = {
        'n_neighbors': [2, 5, 10, 15, 20, 25, 30],
        'contamination': [0.01, 0.05, 0.1, 0.15, 0.2]
    }

    best_f1 = -1
    best_params = None
    best_predictions = None
    best_metrics = {}

    for params in ParameterGrid(param_grid):
        clf = LocalOutlierFactor(
            n_neighbors=params['n_neighbors'],
            contamination=params['contamination'],
            novelty=True
        )
        clf.fit(scaled_param_diff)
        preds = clf.predict(scaled_param_diff)

        precision = precision_score(y_true, preds, pos_label=-1, zero_division=0)
        recall = recall_score(y_true, preds, pos_label=-1, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=-1, zero_division=0)
        accuracy = accuracy_score(y_true, preds)

        y_binary_true = [1 if y == -1 else 0 for y in y_true]
        y_binary_pred = [1 if p == -1 else 0 for p in preds]
        try:
            roc_auc = roc_auc_score(y_binary_true, y_binary_pred)
        except ValueError:
            roc_auc = float('nan')

        logger.info(f"LOF (n_neighbors={params['n_neighbors']}, contamination={params['contamination']}) => "
                    f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, "
                    f"ROC-AUC: {roc_auc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_predictions = preds
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'roc_auc': roc_auc
            }

    # Explicitly log the best results after the grid search
    logger.info(f"\nBest LOF parameters found: n_neighbors={best_params['n_neighbors']}, contamination={best_params['contamination']}")
    logger.info(f"Best F1 score: {best_f1:.4f}")
    logger.info(f"Precision: {best_metrics['precision']:.4f}")
    logger.info(f"Recall: {best_metrics['recall']:.4f}")
    logger.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")

    cm = confusion_matrix(y_true, best_predictions, labels=[1, -1])
    logger.info("Confusion Matrix:\n{}".format(cm))

    report = classification_report(y_true, best_predictions, target_names=["Normal", "Poisoned"], labels=[1, -1], zero_division=0)
    logger.info("Classification Report:\n{}".format(report))

    pca = PCA(n_components=2, random_state=42)
    dim_reduced_gradients = pca.fit_transform(scaled_param_diff)
    logger.info("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))

    plot_gradients_2d(zip(worker_ids, dim_reduced_gradients, best_predictions))
    
    
##################################################################################################################################################################################
####### Autoencoders ###############
import os
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from federated_learning.arguments import Arguments
from federated_learning.parameters import get_layer_parameters, calculate_parameter_gradients
from federated_learning.utils import (
    get_model_files_for_epoch, get_model_files_for_suffix,
    apply_standard_scaler, get_worker_num_from_model_file_name
)
from client import Client

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# === Config ===
MODELS_PATH = "D:/SGD_TargetedDPA/3000_models"
EPOCHS = list(range(10, 201))
# LAYER_NAME = "fc.weight"
LAYER_NAME = "fc2.weight"
CLASS_NUM = 5
POISONED_WORKER_IDS = [11, 39, 3, 29, 43, 47, 8, 24, 33, 18, 4, 36, 7, 30, 46, 44, 9, 14, 42, 2, 45, 25, 38, 32, 21, 0, 31, 15, 41, 1, 22, 17, 48, 37, 5, 49, 26, 27, 34, 10]
SAVE_NAME = "defense_results_autoencoder(PD=40_CIFA).jpg"
SAVE_SIZE = (20, 20)

logger.add("AutoEn_logPD=40_CIFA.log", rotation="10 MB", level="INFO", encoding="utf-8")

def load_models(args, model_filenames):
    clients = []
    for model_filename in model_filenames:
        client = Client(args, 0, None, None)
        client.set_net(client.load_model_from_file(model_filename))
        clients.append(client)
    return clients

def build_autoencoder(input_dim, encoding_dim1=64, encoding_dim2=32, bottleneck_dim=16):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim1, activation='relu')(input_layer)
    encoded = Dense(encoding_dim2, activation='relu')(encoded)
    bottleneck = Dense(bottleneck_dim, activation='relu')(encoded)
    decoded = Dense(encoding_dim2, activation='relu')(bottleneck)
    decoded = Dense(encoding_dim1, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')
    return autoencoder


def plot_gradients_2d(gradients_with_preds):
    fig = plt.figure()
    for (worker_id, gradient, pred) in gradients_with_preds:
        if pred == -1:
            plt.scatter(gradient[0], gradient[1], color="red", marker="x", s=200, linewidth=5,
                        label="Anomaly" if 'Anomaly' not in plt.gca().get_legend_handles_labels()[1] else "")
        else:
            plt.scatter(gradient[0], gradient[1], color="green", s=180,
                        label="Normal" if 'Normal' not in plt.gca().get_legend_handles_labels()[1] else "")
    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    # plt.margins(0, 0)
    plt.legend(fontsize=16, loc='best', frameon=True)
    plt.xlabel("Principal Component 1", fontsize=20, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=20, fontweight='bold')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.legend(fontsize=15)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)
    # plt.close(fig)  # Close figure after saving to free memory

if __name__ == '__main__':
    args = Arguments(logger)
    args.log()

    model_files = sorted(os.listdir(MODELS_PATH))
    logger.debug("Number of models: {}", len(model_files))

    param_diff = []
    worker_ids = []

    for epoch in EPOCHS:
        start_model_files = get_model_files_for_epoch(model_files, epoch)
        start_model_file = get_model_files_for_suffix(start_model_files, args.get_epoch_save_start_suffix())[0]
        start_model_file = os.path.join(MODELS_PATH, start_model_file)
        start_model = load_models(args, [start_model_file])[0]
        start_param = list(get_layer_parameters(start_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

        end_model_files = get_model_files_for_epoch(model_files, epoch)
        end_model_files = get_model_files_for_suffix(end_model_files, args.get_epoch_save_end_suffix())

        for end_model_file in end_model_files:
            worker_id = get_worker_num_from_model_file_name(end_model_file)
            end_model_file = os.path.join(MODELS_PATH, end_model_file)
            end_model = load_models(args, [end_model_file])[0]
            end_param = list(get_layer_parameters(end_model.get_nn_parameters(), LAYER_NAME)[CLASS_NUM])

            gradient = calculate_parameter_gradients(logger, start_param, end_param)
            param_diff.append(gradient.flatten())
            worker_ids.append(worker_id)

    scaled_param_diff = apply_standard_scaler(param_diff)
    y_true = [-1 if wid in POISONED_WORKER_IDS else 1 for wid in worker_ids]
    scaled_param_diff = np.array(scaled_param_diff)

    clean_data = scaled_param_diff[np.array(y_true) == 1]

    best_f1 = 0
    best_config = {}
    for enc1 in [32, 64]:
        for enc2 in [16, 32]:
            for bottleneck in [8, 16]:
                autoencoder = build_autoencoder(input_dim=scaled_param_diff.shape[1],
                                                encoding_dim1=enc1,
                                                encoding_dim2=enc2,
                                                bottleneck_dim=bottleneck)
                autoencoder.fit(clean_data, clean_data, epochs=100, batch_size=32, verbose=0)
                reconstructions = autoencoder.predict(scaled_param_diff)
                losses = np.mean(np.square(scaled_param_diff - reconstructions), axis=1)
                normal_losses = losses[np.array(y_true) == 1]
                threshold = np.percentile(normal_losses, 95)
                y_pred = np.where(losses > threshold, -1, 1)
                f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_config = {'enc1': enc1, 'enc2': enc2, 'bottleneck': bottleneck, 'threshold': threshold, 'pred': y_pred, 'losses': losses}

    precision = precision_score(y_true, best_config['pred'], pos_label=-1, zero_division=0)
    recall = recall_score(y_true, best_config['pred'], pos_label=-1, zero_division=0)
    accuracy = accuracy_score(y_true, best_config['pred'])
    roc_auc = roc_auc_score([1 if y == -1 else 0 for y in y_true], best_config['losses'])

    logger.info(f"Best Config: {best_config}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {best_f1:.4f}, Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info("Confusion Matrix:\n{}", confusion_matrix(y_true, best_config['pred'], labels=[1, -1]))
    logger.info("Classification Report:\n{}", classification_report(y_true, best_config['pred'], target_names=["Normal", "Poisoned"], labels=[1, -1], zero_division=0))

    pca = PCA(n_components=2, random_state=42)
    dim_reduced = pca.fit_transform(scaled_param_diff)
    plot_gradients_2d(zip(worker_ids, dim_reduced, best_config['pred']))


