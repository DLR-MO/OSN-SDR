# SPDX-FileCopyrightText: 2023 Emy Arts <emy.arts@dlr.de>
# SPDX-FileCopyrightText: 2023 German Aerospace Center
#
# SPDX-License-Identifier: MIT

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import numpy as np
from pathlib import Path
import argparse
from sklearn.utils import resample
from joblib import dump, load
import random
from sklearn.utils import shuffle
from prepare_datasets import DS_PATH

RESULT_PATH = DS_PATH.parent.parent / "results"
(RESULT_PATH / "svm_models").mkdir(exist_ok=True, parents=True)
(RESULT_PATH / "svm_results").mkdir(exist_ok=True, parents=True)


def svm(max_iters, aimed_insp_rate=0.1, c=25, gamma=0.1, class_weight='balanced', trial=''):
    features_df = pd.read_csv(DS_PATH / "svm_train_set.csv").dropna(axis=0)

    n = features_df.shape[0]
    mask = np.full(n, True)
    rand = random.sample(range(n), int(n * 0.1))
    mask[rand] = False
    train_df = features_df[mask]
    val_df = features_df[~mask]
    if aimed_insp_rate != 0:
        no_insp_df = train_df[train_df["Inspection"] == 0]
        insp_df = train_df[train_df["Inspection"] == 1]
        insp_rate = insp_df.shape[0] / no_insp_df.shape[0]
        print("Pre resampling inspection rate", insp_rate)
        no_insp_n_samples = int(train_df.shape[0] * (1-aimed_insp_rate))
        insp_n_samples = int(train_df.shape[0] * (aimed_insp_rate))
        no_insp_features_df_upsampled = resample(no_insp_df,
                                              replace=False,
                                              n_samples=no_insp_n_samples)
        insp_features_df_upsampled = resample(insp_df,
                                                 replace=True,
                                                 n_samples=insp_n_samples)
        train_df = pd.concat([no_insp_features_df_upsampled, insp_features_df_upsampled])
    print("Inspection Rate", train_df["Inspection"].sum() / train_df.shape[0])

    x_train, y_train = shuffle(train_df.drop(columns="Inspection").values, train_df["Inspection"].values)
    x_val, y_val = shuffle(val_df.drop(columns="Inspection").values, val_df["Inspection"].values)

    print(f"Training SVM ({len(x_train)})")
    clf = SVC(verbose=True, max_iter=max_iters, class_weight='balanced', C=c, gamma=gamma)
    clf.fit(x_train, y_train)
    print(f"Testing SVM with {class_weight} weight, {aimed_insp_rate} resampled inspection rate, {max_iters} iterations")
    preds = clf.predict(x_val)
    print(f"{preds.sum()} predicted inspections {y_val.sum()} actual inspections")
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    bacc = balanced_accuracy_score(y_val, preds)

    print("Accuracy", acc)
    print("Balanced Accuracy", bacc)
    print("F1", f1)

    results_df = pd.DataFrame({"InspectionClassWeight": [class_weight],
                               "ResamplingInspectionRate": [aimed_insp_rate],
                               "Gamma": [gamma],
                               "C": [c],
                               "Iterations": [max_iters],
                               "F1Score": [f1],
                               "Accuracy": [acc],
                               "BalancedAccuracy": [bacc]
    })
    file_name = f"t{trial}_r{int(aimed_insp_rate * 10)}_i{int(max_iters / 1000)}k_g{gamma}_c{c}_w{class_weight}_a{int(acc * 100)}_f{int(f1 * 100)}_ba{int(bacc * 100)}"
    dump(clf, RESULT_PATH / "svm_models" / f"{file_name}.joblib")
    results_df.to_csv(RESULT_PATH / "svm_results" / f"{file_name}.csv", index=False)
    return results_df

#%%
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a parameter gridsearch of the flight phase estimating network")
    parser.add_argument("--r", default=0.1, type=float, help="Inspection rate after resampling")
    parser.add_argument("--iter", default=7000, type=int, help="Maximum number of iterations")
    parser.add_argument("--c", default=25, type=float, help="SVM C regularisation parameter")
    parser.add_argument("--gamma", default=0.1, type=float, help="SVM gamma regularisation parameter")
    args = parser.parse_args()
    aimed_insp_rate = args.r
    MAX_ITERS = args.iter
    c = args.c
    gamma = args.gamma

    results = svm(aimed_insp_rate=aimed_insp_rate, max_iters=MAX_ITERS, c=c, gamma=gamma)
