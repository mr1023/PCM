import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import sklearn
import mglearn


data = pd.read_excel('01 Q4_21_Q1_22.xlsx', sheet_name="PCA", engine='openpyxl')
print()

vars = len(data.columns)

data.info()
print("Información de los datos:")
print(data.columns)
print()

summary = data.describe()
print('Resumen:')
print(summary)

# Correlación entre las variables
cor = data.corr()
print("Correlación entre variables:")


# Graficar correlacion por mapa de calor
plt.figure()
sns.heatmap(cor,annot = True,
            xticklabels = cor.columns,
            yticklabels = cor.columns,
            cmap = 'coolwarm')
plt.show()

# Escalamiento de datos entre [0,1]; este método permite presencia de outliers
scaler = MinMaxScaler()
scaler.fit(data)
scaled_dataMMS = scaler.transform(data)
print("Escalamiento de datos:")
print(scaled_dataMMS)
print()

#Algoritmo pca
pca = PCA()
pca = PCA(n_components=vars) # Indicar el No. de componentes ppales
pca.fit(scaled_dataMMS)

# Ponderación de los componentes principales
pca_score=pd.DataFrame(data = pca.components_, columns = data.columns,)

# Mapa de calor para visualizar in influencia de las variables
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
componentes = pca.components_
plt.imshow(componentes.T, cmap='plasma', aspect='auto')
plt.yticks(range(len(data.columns)), data.columns)
plt.xticks(range(len(data.columns)), np.arange(pca.n_components_) + 1)
plt.grid(False)
plt.colorbar()
plt.show()

# Gráfica del aporte a cada componente principal
# Aporte al primer componente principal
matrix_transform = pca.components_.T
plt.bar(np.arange(vars),matrix_transform[:,0])
plt.xlabel('Núm. variable')
plt.ylabel('Vector asociado')
plt.show()

# Obtener las primeras 10 variables con mayor aporte
loading_scores = pd.DataFrame(pca.components_[0])
loading_scores.index = data.columns
sorted_loading_scores = loading_scores[0].abs().sort_values(ascending=False)
top_variables = sorted_loading_scores[0:10].index.values
print("Scores:")
print(sorted_loading_scores[0:10])
print()
print("Top 10 variables:")
print(top_variables)
print()


# Porcentaje de varianza explicada por cada componente principal proporciona
# Lambda/suma_Lambda (valor_propio/suma_valores_propios)
per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)
porcent_acum = np.cumsum(per_var) # % varianza acumulada de los componentes
print("Porcentaje:")
print(porcent_acum)
print()
