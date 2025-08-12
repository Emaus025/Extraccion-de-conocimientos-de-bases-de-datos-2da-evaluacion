import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Cargar los datos
df = pd.read_csv('beisbol.csv')

# Exploración inicial
print("Información del dataset:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe())

# Visualizar la relación entre bateos y runs    
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bateos', y='runs', data=df)
plt.title('Relación entre Bateos y Runs')
plt.xlabel('Bateos')
plt.ylabel('Runs')
plt.savefig('beisbol_scatter.png')
plt.close()

# Preparar los datos para el modelo
X = df[['bateos']]
y = df['runs']

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResultados del modelo:")
print(f"Error cuadrático medio (MSE): {mse:.2f}")
print(f"Coeficiente de determinación (R²): {r2:.2f}")
print(f"Coeficiente (pendiente): {model.coef_[0]:.4f}")
print(f"Intercepto: {model.intercept_:.2f}")

# Visualizar los resultados del modelo
plt.figure(figsize=(10, 6))

# Datos reales
plt.scatter(X_test, y_test, color='blue', label='Datos reales')

# Línea de regresión
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicción')

plt.title('Modelo de Regresión Lineal: Bateos vs Runs')
plt.xlabel('Bateos')
plt.ylabel('Runs')
plt.legend()
plt.grid(True)
plt.savefig('beisbol_regresion.png')
plt.close()

# Análisis de residuos
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribución de Residuos')
plt.xlabel('Residuos')
plt.ylabel('Frecuencia')
plt.savefig('beisbol_residuos.png')
plt.close()

# Guardar el modelo
import joblib
joblib.dump(model, 'modelo_beisbol.pkl')
print("\nModelo guardado como 'modelo_beisbol.pkl'")