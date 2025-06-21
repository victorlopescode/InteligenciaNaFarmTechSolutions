from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('dados.csv')
X = df[['umidade', 'nutrientes', 'hora']]
y = df['precisa_irrigar']

X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Exibir taxa de acurácia do modelo
acuracia = model.score(X_test, y_test)
print(f"Acurácia do modelo: {acuracia:.2%}")

# Gerar amostras aleatórias para previsão
num_amostras = 5
amostras = pd.DataFrame({
    'umidade': np.random.uniform(10, 80, num_amostras),
    'nutrientes': np.random.uniform(20, 100, num_amostras),
    'hora': np.random.randint(0, 24, num_amostras)
})
previsoes = model.predict(amostras)
for i, row in amostras.iterrows():
    print(f"Horário: {int(row['hora'])}h | Umidade: {row['umidade']:.2f} | Nutrientes: {row['nutrientes']:.2f} => Precisa irrigar: {previsoes[i]}")

joblib.dump(model, 'modelo_irrigacao.pkl')  # Salva o modelo treinado
