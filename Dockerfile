# Usa uma imagem leve do Python
FROM python:3.11-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências
COPY requirements.txt .

# Instala as bibliotecas
RUN pip install --no-cache-dir -r requirements.txt

# Copia o seu script python para dentro do container
COPY oracle.py .

# Expõe a porta padrão do Streamlit para acesso externo
EXPOSE 8501

# Comando para rodar o servidor web do Streamlit
CMD ["streamlit", "run", "oracle.py", "--server.port=8501", "--server.address=0.0.0.0"]