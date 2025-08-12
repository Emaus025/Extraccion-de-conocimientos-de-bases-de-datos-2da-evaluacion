import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from datetime import datetime

# Cargar los datos
df = pd.read_csv('samsung.csv')

# Convertir la columna de fecha a datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Exploración inicial
print("Información del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Crear características adicionales
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Calcular características de tendencia
df['Close_Lag1'] = df['Close'].shift(1)
df['Volume_Lag1'] = df['Volume'].shift(1)
df['Close_Change'] = df['Close'] - df['Close_Lag1']
df['Volume_Change'] = df['Volume'] - df['Volume_Lag1']

# Eliminar filas con NaN (primera fila)
df = df.dropna()

# Seleccionar características para clustering
features = ['Close', 'Volume', 'Close_Change', 'Volume_Change']
X = df[features]

# Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinar el número óptimo de clusters usando el método del codo
inertia = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Graficar el método del codo
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouette_scores, 'o-')
plt.xlabel('Número de clusters')
plt.ylabel('Puntuación de Silueta')
plt.title('Puntuación de Silueta vs. Número de Clusters')
plt.grid(True)

plt.tight_layout()
plt.savefig('samsung_elbow_method.png')
plt.close()

# Seleccionar el número óptimo de clusters (basado en los gráficos anteriores)
optimal_k = 4  # Ajustar según los resultados del método del codo y silueta

# Aplicar K-means con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizar los clusters en 2D usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis', s=100, alpha=0.7)
plt.title(f'Visualización de Clusters (K={optimal_k}) usando PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.savefig('samsung_clusters_pca.png')
plt.close()

# Analizar características de cada cluster
cluster_analysis = df.groupby('Cluster')[features].mean()
print("\nCaracterísticas promedio por cluster:")
print(cluster_analysis)

# Visualizar la distribución de precios de cierre por cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='Close', data=df, palette='viridis')
plt.title('Distribución de Precios de Cierre por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Precio de Cierre')
plt.savefig('samsung_close_by_cluster.png')
plt.close()

# Visualizar la evolución temporal de los clusters
plt.figure(figsize=(14, 8))
sns.scatterplot(x='Date', y='Close', hue='Cluster', data=df, palette='viridis', s=100, alpha=0.7)
plt.title('Evolución Temporal de Clusters')
plt.xlabel('Fecha')
plt.ylabel('Precio de Cierre')
plt.legend(title='Cluster')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('samsung_temporal_clusters.png')
plt.close()

# Guardar el modelo y los resultados
import joblib
joblib.dump(kmeans, 'modelo_samsung_kmeans.pkl')
joblib.dump(scaler, 'scaler_samsung.pkl')
df.to_csv('samsung_con_clusters.csv', index=False)
print("\nModelo guardado como 'modelo_samsung_kmeans.pkl'")
print("Datos con clusters guardados como 'samsung_con_clusters.csv'")