import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Cargar los datos
df = pd.read_csv('data/beisbol.csv')

# Exploración inicial
print("Información del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Crear una variable categórica basada en 'runs'
# Clasificaremos los equipos en rendimiento alto, medio o bajo según sus carreras
q_33 = df['runs'].quantile(0.33)
q_66 = df['runs'].quantile(0.66)

def categorize_runs(runs):
    if runs <= q_33:
        return 0  # Rendimiento bajo
    elif runs <= q_66:
        return 1  # Rendimiento medio
    else:
        return 2  # Rendimiento alto

df['rendimiento'] = df['runs'].apply(categorize_runs)

# Visualizar la distribución de clases
plt.figure(figsize=(8, 6))
sns.countplot(x='rendimiento', data=df)
plt.title('Distribución de Clases de Rendimiento')
plt.xlabel('Clase de Rendimiento (0=Bajo, 1=Medio, 2=Alto)')
plt.ylabel('Número de Equipos')
plt.savefig('results/beisbol_distribucion_clases.png')
plt.close()

# Visualizar la relación entre bateos y rendimiento
plt.figure(figsize=(10, 6))
sns.boxplot(x='rendimiento', y='bateos', data=df)
plt.title('Bateos por Clase de Rendimiento')
plt.xlabel('Clase de Rendimiento (0=Bajo, 1=Medio, 2=Alto)')
plt.ylabel('Bateos')
plt.savefig('results/beisbol_bateos_por_rendimiento.png')
plt.close()

# Preparar los datos para el modelo
X = df[['bateos']]
y = df['rendimiento']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de árbol de decisión
tree = DecisionTreeClassifier(random_state=42)

# Definir hiperparámetros para optimización
param_grid = {
    'max_depth': [None, 2, 3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Realizar búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(estimator=tree, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor modelo
best_tree = grid_search.best_estimator_
print(f"\nMejores hiperparámetros: {grid_search.best_params_}")

# Evaluar el modelo en el conjunto de prueba
y_pred = best_tree.predict(X_test_scaled)

# Calcular métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}")
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bajo', 'Medio', 'Alto'],
            yticklabels=['Bajo', 'Medio', 'Alto'])
plt.title('Matriz de Confusión - Árbol de Decisión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.savefig('results/beisbol_tree_confusion_matrix.png')
plt.close()

# Visualizar el árbol de decisión
plt.figure(figsize=(15, 10))
plot_tree(best_tree, filled=True, feature_names=['bateos'], 
          class_names=['Bajo', 'Medio', 'Alto'], rounded=True, fontsize=10)
plt.title('Árbol de Decisión para Clasificación de Rendimiento')
plt.savefig('results/beisbol_decision_tree.png')
plt.close()

# Visualizar las fronteras de decisión
def plot_decision_boundaries(X, y, model, title):
    # Crear una malla para visualizar las fronteras de decisión
    h = 0.02  # Tamaño del paso en la malla
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx = np.arange(x_min, x_max, h).reshape(-1, 1)
    
    # Predecir la clase para cada punto en la malla
    Z = model.predict(xx)
    
    # Visualizar los resultados
    plt.figure(figsize=(10, 6))
    
    # Graficar los puntos de datos
    scatter = plt.scatter(X, np.zeros_like(X) + 0.5, c=y, edgecolors='k', 
                         cmap=plt.cm.brg, s=100, alpha=0.8)
    
    # Colorear las regiones según la clase predicha
    plt.pcolormesh(xx.reshape(-1), np.array([0, 1]), 
                  Z.reshape(1, -1), cmap=plt.cm.brg, alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Bateos (Escalados)')
    plt.yticks([])
    plt.legend(*scatter.legend_elements(), title="Clases")
    plt.savefig('results/beisbol_decision_boundaries.png')
    plt.close()

# Visualizar las fronteras de decisión
plot_decision_boundaries(X_train_scaled, y_train, best_tree, 
                        'Fronteras de Decisión - Clasificación de Rendimiento')

# Guardar el modelo
import joblib
joblib.dump(best_tree, 'models/modelo_beisbol_tree.pkl')
joblib.dump(scaler, 'models/scaler_beisbol_tree.pkl')
print("\nModelo guardado como 'modelo_beisbol_tree.pkl'")