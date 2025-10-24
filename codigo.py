import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn

#Carregar dados
arquivo_dados = 'inputs/dados.csv' #Caminho do arquivo
df = pd.read_csv(arquivo_dados)
print(df.head()) #imprime as primeiras linhas

#Criação de gráfico para observar a relação
X = df[['temperatura']]
Y = df['vendas']

plt.figure(figsize=(10,6))
plt.scatter(X, Y, color='purple', alpha=0.7)
plt.title('Relação entre Temperatura e Vendas')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas(unidades)')
plt.grid(True)
plt.show()

#Separando dados de treino e teste
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=42)

#Usando o MLflow
mlflow.set_experiment("Previsao_Vendas_GelatoMagico")

#Criando o modelo de previsão
with mlflow.start_run():
    modelo = LinearRegression()
    modelo.fit(X_treino, Y_treino)

    previsao = modelo.predict(X_teste)

    #Avaliar o modelo
    erro_medio = mean_absolute_error(Y_teste, previsao)
    
    #Registrar no Mlflow
    mlflow.log_param("algoritmo", "LinearRegression")
    mlflow.log_metric("MAE", erro_medio)
    mlflow.sklearn.log_model(modelo, "modelo_vendas")
    print(f"Erro médio (MAE): {erro_medio:.2f}")


plt.figure(figsize=(10,6))
plt.scatter(X,Y, color='purple', alpha=0.7, label='Dados reais')
plt.plot(X_teste, previsao, color='orange', linewidth=2, label='Linha de regressão')
plt.title('Relação entre Temperatura e Vendas (com regressão)')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas (unidades)')
plt.legend()
plt.grid(True)
plt.show()
