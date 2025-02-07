import json
import pandas as pd
import subprocess
from sklearn.feature_selection import mutual_info_classif

ml_features_file = 'data/important_features_ml.json'
prolog_file = "data/mushrooms_kb.pl"

def extract_kb_features():
    query = """
    findall(Features, (mushroom(_, Features)), Results), write(Results), nl.
    """
    try:
        result = subprocess.run([
            "swipl", "-q", "-s", prolog_file, "-g", query, "-t", "halt"],
            capture_output=True, text=True
        )
        output = result.stdout.strip()
        features = set()
        if output:
            output = output.replace("[", "").replace("]", "").replace(" ", "").split("),(")
            for feature_list in output:
                features.update(feature_list.split(","))
        important_features = [feature for feature in features if "_True" in feature]
        with open('data/important_features_kb.json', 'w') as f:
            json.dump(important_features, f, indent=4)
        print("Feature KB estratte e salvate in formato JSON.")
    except Exception as e:
        print(f"Errore nell'estrazione delle feature KB: {e}")

def extract_ml_features():
    df = pd.read_csv('data/mushrooms_encoded.csv')
    X = df.drop(columns=['class_p'])
    y = df['class_p']
    mi_scores = mutual_info_classif(X, y)
    mi_results = pd.DataFrame({'Feature': X.columns, 'Mutual Info Score': mi_scores})
    selected_features = mi_results.sort_values(by="Mutual Info Score", ascending=False).head(20)["Feature"].tolist()
    with open(ml_features_file, 'w') as f:
        json.dump(selected_features, f, indent=4)

    df = pd.read_csv('data/mushrooms_encoded.csv')

    with open('data/feature_comparison.json', 'r') as f:
       feature_comparison = json.load(f)
    common_features = feature_comparison["common_features"]

    X = df[common_features]
    X.to_csv('data/mushrooms_common_features.csv', index=False)

def main():
    extract_kb_features()
    extract_ml_features()

if __name__ == "__main__":
    main()
