import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
df = pd.read_csv('comprar_alquilar.csv')

# Exploración inicial
print("Información del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Separar características y variable objetivo
X = df.drop('comprar', axis=1)
y = df['comprar']

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar PCA
pca = PCA()
pca.fit(X_scaled)

# Visualizar la varianza explicada acumulada
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% de varianza explicada')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada Acumulada')
plt.title('Varianza Explicada Acumulada vs. Número de Componentes')
plt.grid(True)
plt.legend()
plt.savefig('comprar_alquilar_varianza_acumulada.png')
plt.close()

# Determinar el número de componentes para explicar el 95% de la varianza
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"\nNúmero de componentes para explicar el 95% de la varianza: {n_components}")

# Aplicar PCA con el número óptimo de componentes
pca_optimal = PCA(n_components=n_components)
X_pca = pca_optimal.fit_transform(X_scaled)

# Visualizar los datos en las dos primeras componentes principales
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=100, alpha=0.7)
plt.title('Visualización de Datos en las Dos Primeras Componentes Principales')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Comprar', labels=['Alquilar', 'Comprar'])
plt.grid(True)
plt.savefig('comprar_alquilar_pca_visualization.png')
plt.close()

# Analizar la contribución de cada característica a las componentes principales
components = pd.DataFrame(
    pca_optimal.components_,
    columns=X.columns
)

plt.figure(figsize=(14, 10))
sns.heatmap(components, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Contribución de Características a las Componentes Principales')
plt.xlabel('Características')
plt.ylabel('Componentes Principales')
plt.savefig('comprar_alquilar_pca_components.png')
plt.close()

# Evaluar el rendimiento de un modelo con las componentes principales
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar un modelo de clasificación
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo con PCA: {accuracy:.4f}")
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Comparar con el modelo sin reducción de dimensionalidad
X_original_train, X_original_test, y_original_train, y_original_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

rf_original = RandomForestClassifier(random_state=42)
rf_original.fit(X_original_train, y_original_train)
y_original_pred = rf_original.predict(X_original_test)
accuracy_original = accuracy_score(y_original_test, y_original_pred)

print(f"\nPrecisión del modelo sin PCA: {accuracy_original:.4f}")
print("\nInforme de clasificación (sin PCA):")
print(classification_report(y_original_test, y_original_pred))

# Visualizar la matriz de correlación de las características originales
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación de Características')
plt.tight_layout()
plt.savefig('comprar_alquilar_correlation_matrix.png')
plt.close()

# Guardar el modelo y los transformadores
import joblib
joblib.dump(pca_optimal, 'modelo_pca_comprar_alquilar.pkl')
joblib.dump(scaler, 'scaler_comprar_alquilar.pkl')
joblib.dump(rf, 'modelo_rf_pca_comprar_alquilar.pkl')
print("\nModelo PCA guardado como 'modelo_pca_comprar_alquilar.pkl'")
print("Modelo RandomForest con PCA guardado como 'modelo_rf_pca_comprar_alquilar.pkl'")