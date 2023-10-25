import pandas as pd
from pathlib import Path
from joblib import load
from plot_utils import show_svm_decision_function
from sklearn.metrics import accuracy_score,  balanced_accuracy_score
from svm import RESULT_PATH

#%%
result_dfs = []
for file in (RESULT_PATH / "svm_results").iterdir():
    result_dfs.append(pd.read_csv(file))
result_df = pd.concat(result_dfs).reset_index(drop=True)
for col in ["Iterations", "Accuracy", "F1Score", "BalancedAccuracy"]:
    result_df.loc[:, col] = pd.to_numeric(result_df[col])

#%%
test_df = pd.read_csv(RESULT_PATH / "svm_test_set.csv")
test_x = test_df.drop(columns="Inspection").values
test_y = test_df["Inspection"].values

#%%
model_folder = Path.cwd() / "results" / "svm_models"
model_name = "r1_i7k_g0.1_c25_wbalanced_a86_f1_ba77.joblib"
clf = load(model_folder / model_name)
pred = clf.predict(test_x)

print("Accuracy", accuracy_score(test_y, pred))
print("BalancedAccuracy", balanced_accuracy_score(test_y, pred))

show_svm_decision_function(clf, save="DecisionFunction.pdf")