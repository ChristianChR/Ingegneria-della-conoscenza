import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

os.makedirs("images", exist_ok=True)

df = pd.read_csv('data/mushrooms_common_features.csv')
y = pd.read_csv('data/mushrooms_encoded.csv')['class_p']

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)

best_params = {}
with open("data/best_hyperparameters.txt", "r") as f:
    lines = f.readlines()
    current_model = None
    for line in lines:
        line = line.strip()
        if line.endswith(":"):
            current_model = line[:-1]
            best_params[current_model] = {}
        elif current_model and line:
            param, value = line.split(": ")
            try:
                best_params[current_model][param] = eval(value)
            except (NameError, SyntaxError):
                best_params[current_model][param] = value.strip("\"'")

rf = RandomForestClassifier(**best_params["random_forest"], random_state=42)
svm = SVC(**best_params["svm"], probability=True)
dt = DecisionTreeClassifier(**best_params["decision_tree"], random_state=42)
nb = BernoulliNB()  # Modello Naive Bayes

kf = KFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "Random Forest": rf,
    "SVM": svm,
    "Decision Tree": dt,
    "Naive Bayes": nb
}

for name, model in models.items():
    print(f"Training {name} with K-Fold Cross Validation...")
    scores = []
    for train_idx, val_idx in kf.split(X_train):
        model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        score = model.score(X_train.iloc[val_idx], y_train.iloc[val_idx])
        scores.append(score)
    print(f"{name} Mean CV Accuracy: {sum(scores)/len(scores):.4f}")

    print("Addestramento e validazione completati.")
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{name} Test Accuracy: {accuracy:.4f}")
    print(f"{name} Test Precision: {precision:.4f}")
    print(f"{name} Test Recall: {recall:.4f}")
    print(f"{name} Test F1 Score: {f1:.4f}")
    
    # ROC Curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc="lower right")
    plt.savefig(f"images/{name.replace(' ', '_')}_roc_curve.png")
    plt.close()
    
    train_errors = []
    test_errors = []
    train_sizes = [int(len(X_train) * x) for x in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]
    
    for size in train_sizes:
        X_partial, y_partial = X_train[:size], y_train[:size]
        model.fit(X_partial, y_partial)
        train_error = 1 - model.score(X_partial, y_partial)
        test_error = 1 - model.score(X_test, y_test)
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    # Calcolo della varianza e deviazione standard degli errori
    train_variances = np.var(train_errors)
    test_variances = np.var(test_errors)
    train_std_devs = np.std(train_errors)
    test_std_devs = np.std(test_errors)
    
    plt.figure()
    plt.plot(train_sizes, train_errors, label="Train Error")
    plt.plot(train_sizes, test_errors, label="Test Error")
    plt.fill_between(train_sizes, np.array(train_errors) - np.array(train_std_devs), np.array(train_errors) + np.array(train_std_devs), alpha=0.2)
    plt.fill_between(train_sizes, np.array(test_errors) - np.array(test_std_devs), np.array(test_errors) + np.array(test_std_devs), alpha=0.2)
    plt.xlabel("Training Set Sizes")
    plt.ylabel("Error")
    plt.title(f"Learning Curve - {name}")
    plt.legend()
    plt.savefig(f"images/{name.replace(' ', '_')}_learning_curve.png")
    plt.close()


    test_data = pd.DataFrame([
        {   # Agaricus bisporus (edibile)
            "cap-shape_x": True, "cap-surface_s": True, "cap-color_b": True, "bruises_t": False,
            "odor_f": False, "gill-color_p": True, "stalk-shape_e": True, "ring-type_p": True,
            "spore-print-color_b": True, "population_s": True, "habitat_g": True
        },
        {  # Amanita phalloides (velenoso)
            "cap-shape_x": True, "cap-surface_s": True, "cap-color_g": True, "bruises_t": False,
            "odor_p": True, "gill-color_w": True, "stalk-shape_t": True, "ring-type_p": True,
            "spore-print-color_w": True, "population_s": True, "habitat_f": True
        }
    ])

    full_columns = pd.read_csv("data/mushrooms_encoded.csv").columns
    test_data = test_data.reindex(columns=full_columns, fill_value=False)

    test_data = test_data.astype(bool)

    for name, model in models.items():
        predictions = model.predict(test_data)
        print(f"{name} Predictions: {predictions}")