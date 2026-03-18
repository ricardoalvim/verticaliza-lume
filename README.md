# verticaliza-lume
Linguagem, Urbanismo, Memória e Engenharia

# Build sem cache para forçar instalação limpa das bibliotecas
# docker build --no-cache -t verticaliza-lume .

# Build COM cache - mais rápido!
docker build -t verticaliza-lume .

# Run Interativo com seu arquivo .env
docker run -it --rm --env-file .env verticaliza-lume

# Guardrails (anti-alucinação)
- Por padrão, o modo é **base-only**: se a resposta não estiver nos dados do Hygraph, o LUME responde **"Não consta na base."**
- Para habilitar o LLM explicitamente (menos seguro), adicione `ALLOW_LLM=1` no `.env`