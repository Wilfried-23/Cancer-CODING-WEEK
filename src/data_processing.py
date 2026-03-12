import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Récupération de la base de données
# Dataset source: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors [cite: 16]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv"
df = pd.read_csv(url, na_values="?") 

# Définition de la cible (Target)
# Le dataset contient plusieurs tests (Hinselmann, Schiller, Cytology, Biopsy)
# Généralement, on définit le risque si l'un des tests est positif [cite: 24]
df['target'] = df[['Hinselmann', 'Schiller', 'Citology', 'Biopsy']].max(axis=1)
X = df.drop(columns=['Hinselmann', 'Schiller', 'Citology', 'Biopsy', 'target'])
y = df['target']

# 2. Division en 80% entraînement et 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# 3. Retrait des colonnes avec >= 60% de valeurs manquantes
threshold_col = 0.6 * len(X_train)
X_train = X_train.dropna(thresh=threshold_col, axis=1)
X_test = X_test[X_train.columns] # Aligner le test sur les colonnes conservées 

# 4. Retrait des lignes avec >= 60% de valeurs manquantes
threshold_row = 0.6 * X_train.shape[1]
mask_train = X_train.isnull().sum(axis=1) < (X_train.shape[1] - threshold_row)
X_train = X_train[mask_train]
y_train = y_train[mask_train]

# 5. Détection et suppression des valeurs aberrantes (IQR) - CORRIGÉ
# On filtre uniquement les colonnes numériques non-binaires pour ne pas supprimer les 0/1 
num_cols = X_train.select_dtypes(include=[np.number]).columns
for col in num_cols:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR > 0: # Évite d'éliminer les colonnes constantes ou binaires 
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filtrage
        mask = (X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)
        X_train = X_train[mask]
        y_train = y_train[mask]

# 6. Remplacement des valeurs manquantes par la médiane
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 7. Matrice de corrélation (Analyse visuelle recommandée dans le notebook) [cite: 35]
def plot_correlation(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title("Matrice de Corrélation des Caractéristiques")
    plt.show()

# 8. Gestion du déséquilibre avec SMOTE - SÉCURISÉ 
# On ajuste k_neighbors pour qu'il soit inférieur au nombre d'échantillons de la classe minoritaire
min_samples = y_train.value_counts().min()
k_neigh = min(5, min_samples - 1) if min_samples > 1 else 1

smote = SMOTE(k_neighbors=k_neigh, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train)

# 9. Normalisation des valeurs
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_resampled)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"Forme finale X_train: {X_train_scaled.shape}")
print(f"Distribution des classes après SMOTE: {np.bincount(y_resampled.astype(int))}")