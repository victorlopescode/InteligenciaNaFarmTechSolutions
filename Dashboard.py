import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Carregar dados
try:
    df = pd.read_csv("dados.csv")
except Exception as e:
    st.error("Erro ao ler o arquivo dados.csv. Verifique se o arquivo existe e está no formato correto.")
    st.stop()

# Validação simples
def validar_dados(df):
    return not df.empty and all(col in df.columns for col in ['umidade', 'nutrientes', 'hora', 'precisa_irrigar'])

if not validar_dados(df):
    st.error("Arquivo CSV inválido ou colunas ausentes. Verifique o formato.")
    st.stop()

# Carregar modelo treinado
try:
    modelo = joblib.load('modelo_irrigacao.pkl')
except Exception as e:
    st.error("Modelo não encontrado ou erro ao carregar. Treine e salve o modelo com scikit-learn.py.")
    st.stop()

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

# Gráfico de linha para umidade, nutrientes e hora
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df['umidade'], label='Umidade', color='blue', marker='o', alpha=0.6)
ax.plot(df.index, df['nutrientes'], label='Nutrientes', color='green', marker='s', alpha=0.6)
ax.plot(df.index, df['hora'], label='Hora', color='orange', marker='^', alpha=0.4)
ax.set_xlabel('Leitura')
ax.set_ylabel('Valor')
ax.set_title('Umidade, Nutrientes e Hora das Leituras')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.3)
st.pyplot(fig)

# Gráfico de barras para previsões de irrigação
st.markdown('#### Distribuição da Necessidade de Irrigação')
fig2, ax2 = plt.subplots()
contagem = df['precisa_irrigar'].value_counts().sort_index()
ax2.bar(['Não', 'Sim'], contagem, color=['#4CAF50', '#F44336'])
ax2.set_ylabel('Quantidade')
ax2.set_title('Necessidade de Irrigação (0=Não, 1=Sim)')
for i, v in enumerate(contagem):
    ax2.text(i, v + 2, str(v), ha='center', fontweight='bold')
st.pyplot(fig2)
