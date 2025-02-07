import json
import pandas as pd
import matplotlib.pyplot as plt

with open('data/important_features_kb.json', 'r') as f:
    kb_features = set(json.load(f))

with open('data/important_features_ml.json', 'r') as f:
    ml_features = set(json.load(f))

kb_features_normalized = set(f.replace('_True', '') for f in kb_features)

common_features = kb_features_normalized.intersection(ml_features)
only_kb = kb_features_normalized - ml_features
only_ml = ml_features - kb_features_normalized

print(f"Feature comuni: {len(common_features)}")
print(f"Feature solo in KB: {len(only_kb)}")
print(f"Feature solo in ML: {len(only_ml)}")

labels = ['Comuni', 'Solo KB', 'Solo ML']
values = [len(common_features), len(only_kb), len(only_ml)]

plt.figure(figsize=(8,5))
plt.bar(labels, values)
plt.ylabel('Numero di Feature')
plt.title('Confronto Feature KB vs ML')
plt.savefig('images/feature_comparison.png')
plt.show()

comparison_results = {
    "common_features": list(common_features),
    "only_in_kb": list(only_kb),
    "only_in_ml": list(only_ml)
}

with open('data/feature_comparison.json', 'w') as f:
    json.dump(comparison_results, f, indent=4)

print("Confronto completato. Risultati salvati in 'data/feature_comparison.json' e grafico generato.")