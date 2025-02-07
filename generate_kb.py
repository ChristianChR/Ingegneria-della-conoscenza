import pandas as pd

df = pd.read_csv('data/mushrooms_encoded.csv')

# Lista delle feature più rilevanti basata sul test Chi²
selected_features = [
    "odor_n", "odor_f", "stalk-surface-above-ring_k", "stalk-surface-below-ring_k",
    "gill-color_b", "gill-size_n", "spore-print-color_h", "ring-type_l", "ring-type_p",
    "bruises_t", "spore-print-color_n", "spore-print-color_k", "bruises_f", "gill-spacing_w",
    "population_v", "spore-print-color_w", "gill-size_b", "habitat_p", "stalk-surface-above-ring_s",
    "odor_y"
]

with open('data/mushrooms_kb.pl', 'w') as f:
    f.write('% Definizione dei fatti basati sulle feature selezionate\n')
    for index, row in df.iterrows():
        features = ', '.join([f"{col}_{val}" for col, val in row.items() if col in selected_features])
        f.write(f"mushroom({index + 1}, [{features}]).\n")
    
    f.write('\n% Regole per determinare se un fungo è velenoso o commestibile\n')
    for feature in selected_features:
        f.write(f'isPoisonous(Mushroom) :- mushroom(Mushroom, Features), member({feature}_True, Features).\n')
    
    f.write('\n% Un fungo è edibile se non è velenoso\n')
    f.write('isEdible(Mushroom) :- mushroom(Mushroom, Features), \\+ isPoisonous(Mushroom).\n')
