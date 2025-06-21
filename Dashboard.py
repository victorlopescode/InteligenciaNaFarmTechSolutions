import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("dados.csv")

# Validação simples
if df.empty or not all(col in df.columns for col in ['umidade', 'nutrientes', 'hora', 'precisa_irrigar']):
    st.error("Arquivo CSV inválido ou colunas ausentes. Verifique o formato.")
    st.stop()

# Treinando modelo
X = df[['umidade', 'nutrientes', 'hora']]
y = df['precisa_irrigar']
modelo = RandomForestClassifier()
modelo.fit(X, y)

# Título do Dashboard
st.title(" Dashboard FarmTech - Irrigação Inteligente")

# Painel de simulação
st.markdown("###  Simule uma nova leitura")

col1, col2, col3 = st.columns(3)

with col1:
    umidade = st.slider("Umidade (%)", 0.0, 100.0, 45.0)

with col2:
    nutrientes = st.slider("Nutrientes (%)", 0.0, 100.0, 60.0)

with col3:
    hora = st.slider("Hora do Dia", 0, 23, 14)

entrada = pd.DataFrame({
    'umidade': [umidade],
    'nutrientes': [nutrientes],
    'hora': [hora]
})

previsao = modelo.predict(entrada)[0]

# Resultado da previsão
st.subheader(" Previsão do Modelo")
if previsao == 1:
    st.error("⚠️ Será necessário irrigar!")
else:
    st.success("✅ Não é necessário irrigar.")

# Tabela com últimos dados
st.markdown("### 📋 Leituras recentes")
st.dataframe(df.tail(10))

# Gráficos
st.markdown("### 📈 Gráficos das Variáveis")

# Umidade
st.line_chart(df['umidade'], use_container_width=True)

# Nutrientes
st.line_chart(df['nutrientes'], use_container_width=True)

# Hora
st.line_chart(df['hora'], use_container_width=True)
