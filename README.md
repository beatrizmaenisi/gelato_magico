# GELATO MÁGICO - Previsão de vendas de sorvete 🍦

## Descrição
Realizei o desafio do bootcamp da DIO, no qual tinha o seguinte problema:
"Uma sorveteria chamada Gelato Mágico percebeu que a quantidade de sorvetes vendidos tinha relação com a temperatura. No entanto, sem um planejamento adequado, pode acabar produzindo mais sorvetes do que o necessário e ter prejuízos com desperdícios ou, ao contrário, produzir menos e perder vendas.
Para solucionar esse problema, foi utilizado Machine Learning para prever quantos sorvetes serão vendidos com base na temperatura. Com esse modelo, será possível antecipar a demanda e planejar a produção de maneira eficiente.

## Etapas

1. Coleta dos dados e geração de gráfico para analisar a correlação
2. Criação do modelo preditivo
3. Registro do modelo com MLflow
4. Testando uma nova previsão

## Dependências
As bibliotecas utilizadas foram:
- Pandas
- Matplotlib
- Numpy
- MLflow
- Scikit-learn

obs. Deixei um arquivo 'requirements.txt' que contém todas as dependências com as versões que foram utilizadas. Para instalar faça o seguinte código no terminal(bash):

```
pip install -r requirements.txt
```

## Na hora de rodar 💻
1. Primeiro rodar o arquivo codigo.py, para visualizar os gráficos, treinar o modelo e registrar com MLflow.  
No terminal:
```
python codigo.py
```

2. Após isso, deverá aparecer uma pasta mlruns, com a criação do experimento contendo o modelo salvo. Para visualizar no MLflow faça o seguinte.  
No terminal:
```
mlflow ui
```
Deverá aparecer um link que vai direcionar para o MLflow, mostrando seus experimentos salvos. 

3. Para fazer uma nova previsão, no terminal:
```
python prever.py
```
Irá mostrar para o usuário digitar uma temperatura, gerando como saída a quantidade em unidades do sorvete. 

## Conclusão

Durante o desenvolvimento e testes do projeto foi notável a correlação das variáveis, que conforme a temperatura aumentava, as vendas cresciam quase linearmente. 

O modelo LinearRegression teve um erro médio de 5.91.
O resultado do erro médio significa que, em média, o modelo erra cerca de 6 sorvetes por previsão.
Esse é um resultado muito bom, considerando que os valores variam de cerca de 90 a 180 vendas.



