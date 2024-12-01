# Importações necessárias
import streamlit as st
import numpy as np
import joblib

# Carregar o modelo e o scaler
model = joblib.load('modelo_svm_vencedor.pkl')  # Substitua pelo caminho correto
scaler = joblib.load('scaler_svm.pkl')  # Substitua pelo caminho correto

# Título da aplicação
st.title("Previsão da Taxa de Esgotamento (Burn Rate)")

# Criar sliders para as variáveis do dataset
mental_fatigue_score = st.slider('Mental Fatigue Score', 0.0, 10.0, 5.0)  # Exemplo de valor entre 0 e 10
resource_allocation = st.slider('Resource Allocation', 1.0, 10.0, 5.0)  # Valor entre 1 e 10
designation = st.slider('Designation', 0.0, 5.0, 2.5)  # Valor entre 0 e 5

# Criar seletores para variáveis categóricas que foram convertidas em dummies
gender = st.selectbox('Gender', ['Male', 'Female'])
company_type = st.selectbox('Company Type', ['Product', 'Service'])
wfh_setup_available = st.selectbox('WFH Setup Available', ['Yes', 'No'])

# Converter as variáveis categóricas para valores numéricos (dummy variables)
gender_dummy = 1 if gender == 'Male' else 0
company_type_dummy = 1 if company_type == 'Product' else 0
wfh_setup_dummy = 1 if wfh_setup_available == 'Yes' else 0

# Inserir os dados do usuário como um array numpy
input_data = np.array([[mental_fatigue_score, resource_allocation, designation, gender_dummy, company_type_dummy, wfh_setup_dummy]])

# Normalizar os dados com o scaler carregado
input_data_scaled = scaler.transform(input_data)

# Fazer a predição usando o modelo carregado
burn_rate_pred = model.predict(input_data_scaled)

# Exibir a predição
st.write(f"A predição da Taxa de Esgotamento (Burn Rate) é: {burn_rate_pred[0]:.2f}")
