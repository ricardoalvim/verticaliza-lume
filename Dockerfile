# Usa uma imagem leve do Python
FROM python:3.11-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências (vamos criar ele abaixo)
COPY requirements.txt .

# Instala as bibliotecas
RUN pip install --no-cache-dir -r requirements.txt

# Copia o seu script python para dentro do container
COPY oracle.py .

# Comando para rodar o script
CMD ["python", "oracle.py"]