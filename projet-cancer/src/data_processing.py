# 1. Chargement des données
import pandas as pd
import numpy as np
df = pd.read_csv('path/to/your/dataset.csv')  # Replace with your actual dataset path

# 2. Division en 80% entraînement et 20% test
from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Retrait des colonnes avec >= 60% de valeurs manquantes
missing_cols = (X_train.isnull().sum() / len(X_train) >= 0.6)
cols_to_drop = X_train.columns[missing_cols].tolist()
X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

# 4. Retrait des lignes avec >= 60% de valeurs manquantes
row_missing = (X_train.isnull().sum(axis=1) / X_train.shape[1] >= 0.6)
X_train = X_train[~row_missing]
y_train = y_train[~row_missing]

# 5. Détection et suppression des valeurs aberrantes (IQR) - CORRIGÉ
def remove_outliers_iqr(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:  # Seulement si IQR > 0 (protège colonnes binaires)
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))
            df_clean.loc[outliers, col] = np.nan
    return df_clean

X_train = remove_outliers_iqr(X_train)

# 6. Remplacement des valeurs manquantes par la médiane
for col in X_train.columns:
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_train[col].median())

# 7. Matrice de corrélation
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 8))
correlation_matrix = pd.concat([X_train, y_train], axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.show()

# 8. Gestion du déséquilibre avec SMOTE - SÉCURISÉ
from imblearn.over_sampling import SMOTE
minority_count = min(sum(y_train == 0), sum(y_train == 1))
k_neighbors = min(minority_count - 1, 5)
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 9. Normalisation des valeurs
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing terminé !")
