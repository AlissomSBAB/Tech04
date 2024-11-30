import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split
import streamlit as st
import plotly.graph_objects as go

# URL da página
url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"

# Tentar extrair tabelas da página
tables = pd.read_html(url)  # Lê todas as tabelas da página

df = tables[2]  # Pega a tabela certa

# Renomeia as colunas
df.columns = df.iloc[0]
df = df[1:].copy()  # Remove a primeira linha que agora é o cabeçalho
df.columns = ['Data', 'Preço']

# Converte os valores de 'Preço'
df['Preço'] = df['Preço'].astype(int) / 100  # Divide por 100 para ajustar

# Converter coluna Data para Datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')

# Renomear as colunas para os nomes padrão exigidos pelo Prophet ('ds' para data, 'y' para valor)
df.rename(columns={'Data': 'ds', 'Preço': 'y'}, inplace=True)

# Dividir os dados em treino (95%) e teste (5%)
train, test = train_test_split(df, test_size=0.05, shuffle=False)  # shuffle=False para manter a ordem temporal

# Instanciar o modelo Prophet
model = Prophet()

# Ajustar o modelo aos dados
model.fit(train)

# Fazer a previsão para os próximos 365 dias
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Ajustando Dfs para Dash
Tabela_de_Histórico = df[['ds', 'y']].head(365)
Tabela_de_Histórico.rename(columns={'ds': 'Data', 'y': 'Preço'}, inplace=True)
Tabela_de_Histórico['Data'] = Tabela_de_Histórico['Data'].dt.date
Tabela_de_Previsões = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(365)
Tabela_de_Previsões.rename(columns={'ds': 'Data', 'yhat': 'Previsão', 'yhat_lower': 'Minimo', 'yhat_upper': 'Maximo'}, inplace=True)
Tabela_de_Previsões['Data'] = Tabela_de_Previsões['Data'].dt.date

# Definir eventos relevantes
eventos = {
    "Ataques na Arábia Saudita (2019)": "2019-09-14",
    "Pandemia de COVID-19 (2020)": "2020-03-01",
    "Conflito Rússia-Ucrânia (2022)": "2022-02-24",
    "Aumento de Produção Saudita (2023)": "2023-07-01"
}

# Criar a página no Streamlit
st.set_page_config(page_title="Previsão do Preço do Petróleo Brent", layout="wide")

# Título
st.title("Previsão do Preço do Petróleo Brent")

# Gráfico de Previsão
st.subheader("Gráfico de Previsão")

# Criar o gráfico com Plotly
fig = go.Figure()

# Adicionar dados históricos
fig.add_trace(go.Scatter(
    x=df[df['ds'] > '2019-01-01']['ds'], 
    y=df[df['ds'] > '2019-01-01']['y'],
    mode='lines', 
    name='Preço Histórico', 
    line=dict(color='blue')
))

# Adicionar previsões
fig.add_trace(go.Scatter(
    x=forecast[forecast['ds'] > '2024-11-19']['ds'], 
    y=forecast[forecast['ds'] > '2024-11-19']['yhat'], 
    mode='lines', 
    name='Preço Previsto', 
    line=dict(color='orange')
))

# Adicionar linhas verticais e anotações para eventos
for evento, data in eventos.items():
    # Linha vertical
    fig.add_trace(go.Scatter(
        x=[data, data],
        y=[df['y'].min(), df['y'].max()],
        mode='lines',
        name=evento,
        line=dict(color='red', dash='dot')
    ))
    # Anotação
    fig.add_annotation(
        x=data,
        y=df['y'].max(),
        text=evento,
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-40
    )

# Configurar layout do gráfico
fig.update_layout(
    title='Previsão do Preço Diário do Petróleo Brent com Eventos Relevantes',
    xaxis_title='Data',
    yaxis_title='Preço (US$)',
    legend_title='Legenda',
    template='plotly_white',
    legend=dict(
        orientation="h",  # Define orientação horizontal
        yanchor="bottom",  # Alinha a legenda pela parte inferior
        y=-0.3,            # Posição vertical (abaixo do gráfico)
        xanchor="center",  # Centraliza horizontalmente
        x=0.5              # Posição horizontal
    )
)


# Exibir o gráfico interativo no Streamlit
st.plotly_chart(fig, use_container_width=True)

# Dividir a página em duas colunas
col1, col2 = st.columns(2)

# Tabela de Histórico na coluna 1
with col1:
    st.subheader("Tabela de Histórico")
    st.dataframe(Tabela_de_Histórico)

# Tabela de Previsões na coluna 2
with col2:
    st.subheader("Tabela de Previsões")
    st.dataframe(Tabela_de_Previsões)