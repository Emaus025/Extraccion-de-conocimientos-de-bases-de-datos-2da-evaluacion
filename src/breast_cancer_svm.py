import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA

# Cargar los datos
df = pd.read_csv('data/breast-cancer.csv')

# Exploración inicial
print("Información del dataset:")
print(df.info())
print("\nDistribución de clases:")
print(df['diagnosis'].value_counts())

# Convertir la variable objetivo a numérica (M=1, B=0)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Eliminar la columna ID que no aporta información para la predicción
df = df.drop('id', axis=1)

# Separar características y variable objetivo
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Escalar las características (muy importante para SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA para reducir dimensionalidad y visualización
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Crear y entrenar el modelo SVM
svm = SVC(probability=True, random_state=42)

# Definir hiperparámetros para optimización
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly']
}

# Realizar búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor modelo
best_svm = grid_search.best_estimator_
print(f"\nMejores hiperparámetros: {grid_search.best_params_}")

# Evaluar el modelo en el conjunto de prueba
y_pred = best_svm.predict(X_test_scaled)

# Calcular métricas de rendimiento
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.4f}")
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benigno (B)', 'Maligno (M)'],
            yticklabels=['Benigno (B)', 'Maligno (M)'])
plt.title('Matriz de Confusión - SVM')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.savefig('results/breast_cancer_svm_confusion_matrix.png')
plt.close()

# Curva ROC
y_prob = best_svm.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Área bajo la curva ROC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - SVM')
plt.legend(loc='lower right')
plt.savefig('results/breast_cancer_svm_roc_curve.png')
plt.close()

# Visualización de la frontera de decisión (usando PCA para reducir a 2D)
def plot_decision_boundary(X, y, model, title):
    h = 0.02  # Tamaño del paso en la malla
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    plt.title(title)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.savefig('results/breast_cancer_svm_decision_boundary.png')
    plt.close()

# Entrenar un modelo SVM con los mejores parámetros en los datos PCA
svm_pca = SVC(kernel=best_svm.kernel, C=best_svm.C, gamma=best_svm.gamma, probability=True)
svm_pca.fit(X_train_pca, y_train)

# Visualizar la frontera de decisión
plot_decision_boundary(X_train_pca, y_train, svm_pca, 'Frontera de Decisión SVM (PCA)')

# Guardar el modelo
import joblib
joblib.dump(best_svm, 'models/modelo_breast_cancer_svm.pkl')
joblib.dump(scaler, 'models/scaler_breast_cancer_svm.pkl')
joblib.dump(pca, 'models/pca_breast_cancer_svm.pkl')
print("\nModelo guardado como 'modelo_breast_cancer_svm.pkl'")