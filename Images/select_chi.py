import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

df_encoded = pd.read_csv('data/mushrooms_encoded.csv')

X = df_encoded.drop(columns=['class_p', 'class_e'])  
y = df_encoded['class_p']  

chi2_selector = SelectKBest(chi2, k=20)  
X_kbest = chi2_selector.fit_transform(X, y)

p_values = chi2_selector.pvalues_
chi2_scores = chi2_selector.scores_

results = pd.DataFrame({'Feature': X.columns, 'Chi2 Score': chi2_scores, 'P-Value': p_values})

results = results.sort_values(by='Chi2 Score', ascending=False)
selected_features = results.head(20)

selected_features.to_csv('data/selected_chi2_features.txt', index=False, sep='\t')
