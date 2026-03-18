# verticaliza-lume
Linguagem, Urbanismo, Memória e Engenharia


# Build sem cache para forçar instalação limpa das bibliotecas
docker build --no-cache -t verticaliza-lume .

# Run Interativo com seu arquivo .env
docker run -it --rm --env-file .env verticaliza-lume