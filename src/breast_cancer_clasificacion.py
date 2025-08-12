import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import SelectFromModel

# Cargar los datos
df = pd.read_csv('breast-cancer.csv')

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

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo de Random Forest
rf = RandomForestClassifier(random_state=42)

# Definir hiperparámetros para optimización
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Realizar búsqueda de hiperparámetros con validación cruzada
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Obtener el mejor modelo
best_rf = grid_search.best_estimator_
print(f"\nMejores hiperparámetros: {grid_search.best_params_}")

# Evaluar el modelo en el conjunto de prueba
y_pred = best_rf.predict(X_test_scaled)

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
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.savefig('breast_cancer_confusion_matrix.png')
plt.close()

# Curva ROC
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Área bajo la curva ROC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.savefig('breast_cancer_roc_curve.png')
plt.close()

# Importancia de características
feature_importances = pd.DataFrame(
    {'feature': X.columns, 'importance': best_rf.feature_importances_}
).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(15))
plt.title('15 Características Más Importantes')
plt.tight_layout()
plt.savefig('breast_cancer_feature_importance.png')
plt.close()

# Selección de características más importantes
sfm = SelectFromModel(best_rf, threshold='median')
sfm.fit(X_train_scaled, y_train)

# Características seleccionadas
selected_features = X.columns[sfm.get_support()]
print(f"\nCaracterísticas seleccionadas ({len(selected_features)}): {', '.join(selected_features)}")

# Guardar el modelo
import joblib
joblib.dump(best_rf, 'modelo_breast_cancer.pkl')
joblib.dump(scaler, 'scaler_breast_cancer.pkl')
print("\nModelo guardado como 'modelo_breast_cancer.pkl'")