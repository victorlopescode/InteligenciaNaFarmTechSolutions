# Sistema de Previsão de Irrigação

Este projeto utiliza dados simulados de umidade, nutrientes e hora do dia para prever a necessidade de irrigação utilizando machine learning (RandomForest) e lógica de decisão.

## Estrutura dos Dados
O arquivo `dados.csv` contém as seguintes colunas:
- **umidade**: valor numérico representando a umidade do solo
- **nutrientes**: valor numérico representando a quantidade de nutrientes
- **hora**: hora do dia (0 a 23)
- **precisa_irrigar**: 1 se é necessário irrigar, 0 caso contrário

A coluna `precisa_irrigar` é calculada pela seguinte lógica:
```python
if umidade < 45 and (hora >= 6 and hora <= 10):
    precisa_irrigar = 1
elif umidade < 35 and nutrientes < 40:
    precisa_irrigar = 1
else:
    precisa_irrigar = 0
```

## Código de Machine Learning (`scikit-learn.py`)
- Lê o arquivo `dados.csv` usando pandas.
- Separa os dados em variáveis de entrada (umidade, nutrientes, hora) e saída (precisa_irrigar).
- Divide os dados em treino e teste.
- Treina um modelo RandomForestClassifier para prever a necessidade de irrigação.
- Exibe a acurácia do modelo.
- Gera amostras aleatórias e faz previsões para diferentes horários do dia, mostrando se é necessário irrigar ou não.

## Exemplo de Saída
```
Acurácia do modelo: 100.00%
Horário: 7h | Umidade: 30.00 | Nutrientes: 50.00 => Precisa irrigar: 1
Horário: 12h | Umidade: 55.00 | Nutrientes: 70.00 => Precisa irrigar: 0
Horário: 18h | Umidade: 40.00 | Nutrientes: 35.00 => Precisa irrigar: 0
```

## Observações
- O modelo atinge alta acurácia porque os dados seguem uma lógica clara e sem ruído.
- Para garantir 100% de acurácia, pode-se usar a própria regra lógica ao invés de machine learning.
- O código pode ser adaptado para receber dados reais de sensores.
