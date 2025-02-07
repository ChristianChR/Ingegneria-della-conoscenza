import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('data/mushrooms_encoded.csv')

X = df.drop(columns=['class_p', 'class_e'])  # Feature set
y = df['class_p']  # Target: 1 = velenoso, 0 = edibile

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    "svm": {
        "C": [0.01, 0.001],  # Aumento valori di regolarizzazione
        "kernel": ["linear", "rbf", "poly"]  # Aggiunto kernel polinomiale
    },
    "random_forest": {
        "n_estimators": [10, 25],  # Aumento numero di alberi
        "max_depth": [1, 2],  # Ridotto ulteriormente max_depth
        "min_samples_split": [3, 5, 10, 15]  # Aumento valori per limitare splitting
    },
    "decision_tree": {
        "max_depth": [1, 2],  # Ridotto ulteriormente max_depth
        "min_samples_split": [3, 5, 10, 15],  # Aumento valori per limitare splitting
        "criterion": ["gini", "entropy"]  # Aggiunto criterio entropy
    }
}

def tune_model(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, params, cv=10, scoring='accuracy', n_jobs=-1)  # Aumento fold a 10
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_

best_params = {}

# Random Forest
rf = RandomForestClassifier(random_state=42)
best_params["random_forest"] = tune_model(rf, param_grid["random_forest"], X_train, y_train)

# SVM
svm = SVC()
best_params["svm"] = tune_model(svm, param_grid["svm"], X_train, y_train)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
best_params["decision_tree"] = tune_model(dt, param_grid["decision_tree"], X_train, y_train)

with open("data/best_hyperparameters.txt", "w") as f:
    for model, params in best_params.items():
        f.write(f"{model}:\n")
        for param, value in params.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")

