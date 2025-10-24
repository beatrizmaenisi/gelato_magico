import mlflow
import numpy as np

# Nome do experimento e caminho do modelo salvo
modelo_path = "mlruns/420714510725972916/models/m-90c1ad1bb0024ad3bff30933f8a3e605/artifacts/"

#Carregar o modelo
modelo = mlflow.sklearn.load_model(modelo_path)

#Entrada do usuário
temperatura = float(input('Digite a temperatura (°C): '))

# Fazer a previsão
previsao = modelo.predict(np.array([[temperatura]]))

print(f"\nTemperatura: {temperatura}°C")
print(f"Previsão de vendas: {previsao[0]:.2f} unidades")