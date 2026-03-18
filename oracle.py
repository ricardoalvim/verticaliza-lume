import os
import sys
import json
import requests
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Força o terminal a lidar com caracteres estranhos
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        pass

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HYGRAPH_URL = os.getenv("HYGRAPH_URL")
HYGRAPH_TOKEN = os.getenv("HYGRAPH_TOKEN")

def sanitize_text(text):
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    clean_text = re.sub(r'[\ud800-\udfff]', '', text)
    return clean_text.encode('utf-8', 'ignore').decode('utf-8')

def safe_json_parse(field, default_val=None):
    if default_val is None:
        default_val = {}
    if not field:
        return default_val
    if isinstance(field, (dict, list)):
        return field
    try:
        return json.loads(field)
    except Exception:
        return default_val

def format_value(val, suffix=""):
    if not val or val == 0 or val == "0":
        return "Não informado"
    return f"{val}{suffix}"

def fetch_hygraph_data():
    headers = {"Authorization": f"Bearer {HYGRAPH_TOKEN}"}
    query = """
    query {
      cities(stage: PUBLISHED, first: 1000) { 
        name state region population area gdp gdpPerCapita averageIncome medianIncome idh geoloc { latitude longitude } infrastructure culture tourism economy state
      }
      constructors(stage: PUBLISHED, first: 1000) { 
        name city companyStatus website foundedYear description address phone email employees crea cnpj
      }
      architects(stage: PUBLISHED, first: 1000) { 
        name city archStatus yearFounded type address phone email website cnpj description specialties team
      }
      condominiums(stage: PUBLISHED, first: 1000) { 
        name slug city type segment buildingStatus address specifications timeline team pricing features historicalNotes units seo
      }
    }
    """
    try:
        response = requests.post(HYGRAPH_URL, json={'query': query}, headers=headers)
        res_json = response.json()
        if 'errors' in res_json:
            print("\n❌ O Hygraph rejeitou a query.")
            return None
        return res_json.get('data', {})
    except Exception as e:
        print(f"❌ Erro ao consultar a base de dados: {e}")
        return None

# =====================================================================
# 🧠 CÉREBRO LOCAL (IN-MEMORY CACHE)
# =====================================================================
class VerticalizaCache:
    """
    Cache em memória altamente otimizado. 
    Gera frases naturais para o LLM não precisar 'pensar' matematicamente,
    economizando tokens absurdamente.
    """
    def __init__(self, data):
        self.cities = {}
        self.constructors = {}
        self.architects = {}
        self.buildings = {}
        
        self._build_cache(data)
        self._calculate_aggregates()

    def _build_cache(self, data):
        for ed in data.get('condominiums', []):
            specs = safe_json_parse(ed.get('specifications'))
            team = safe_json_parse(ed.get('team'))
            
            raw_constructors = team.get('constructors', [])
            cons_names = [c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in raw_constructors]
            
            self.buildings[ed.get('name').lower()] = {
                "nome": ed.get('name'),
                "cidade": ed.get('city'),
                "status": ed.get('buildingStatus', 'Não informado'),
                "andares": specs.get('floors', 0),
                "quartos": specs.get('bedrooms', 0),
                "area": specs.get('unitArea', 0),
                "construtoras": cons_names,
                "arquiteto": team.get('architect', {}).get('name', 'Não informado') if isinstance(team.get('architect'), dict) else 'Não informado'
            }

        for c in data.get('cities', []):
            self.cities[c.get('name').lower()] = {
                "nome": c.get('name'),
                "populacao": c.get('population'),
                "pib": c.get('gdp'),
                "total_edificios": 0 
            }

        for co in data.get('constructors', []):
            self.constructors[co.get('name').lower()] = {
                "nome": co.get('name'),
                "fundacao": co.get('foundedYear', 'Não informado'),
                "total_obras": 0, 
                "obras": []
            }

    def _calculate_aggregates(self):
        for b_key, b_data in self.buildings.items():
            cidade_key = str(b_data['cidade']).lower()
            if cidade_key in self.cities:
                self.cities[cidade_key]['total_edificios'] += 1
            
            for c_name in b_data['construtoras']:
                c_key = str(c_name).lower()
                if c_key in self.constructors:
                    self.constructors[c_key]['total_obras'] += 1
                    self.constructors[c_key]['obras'].append(b_data['nome'])

    def analyze_query(self, query):
        """
        Processamento local transformado em LINGUAGEM NATURAL.
        Evita o tom robótico e a cegueira do Llama-3.
        """
        q_lower = query.lower()
        contexto_local = []

        mencoes_predios = [b_data for b_key, b_data in self.buildings.items() if b_data['nome'].lower() in q_lower]
        
        # Comparações
        if ("compar" in q_lower or "diferença" in q_lower) and len(mencoes_predios) >= 2:
            contexto_local.append("DADOS PARA COMPARAÇÃO:")
            for p in mencoes_predios:
                contexto_local.append(
                    f"O edifício {p['nome']} em {p['cidade']} tem {format_value(p['andares'])} andares, "
                    f"está com status '{p['status']}' e a construtora é {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'}."
                )

        # Agregados de Construtora
        for c_key, c_data in self.constructors.items():
            if c_data['nome'].lower() in q_lower:
                contexto_local.append(
                    f"DADOS SOBRE A CONSTRUTORA {c_data['nome']}: "
                    f"Ela possui {c_data['total_obras']} edifícios registrados na base. "
                    f"Alguns de seus projetos incluem: {', '.join(c_data['obras'][:8])}."
                )

        # Informação Exata do Prédio
        if len(mencoes_predios) == 1 and not "compar" in q_lower:
            p = mencoes_predios[0]
            contexto_local.append(
                f"FICHA TÉCNICA DO EDIFÍCIO {p['nome']}: "
                f"Localizado em {p['cidade']}, possui {format_value(p['andares'])} andares, "
                f"{format_value(p['quartos'])} quartos e área de {format_value(p['area'], 'm²')}. "
                f"Sua construtora é {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'}."
            )

        return "\n".join(contexto_local)

# =====================================================================
# 📚 MOTOR SEMÂNTICO (FAISS) - Enxugado para salvar Tokens
# =====================================================================
def build_documents(data):
    if not data:
        return []
    docs = []

    # Cidades (Apenas os números para evitar as alucinações geográficas)
    for c in data.get('cities', []):
        content = (
            f"DADOS DA CIDADE {c.get('name')} ({c.get('state')}): "
            f"População é {format_value(c.get('population'))}, PIB é {format_value(c.get('gdp'))} e IDH é {format_value(c.get('idh'))}. "
            f"Fica na região {c.get('region')}."
        )
        docs.append(Document(page_content=sanitize_text(content)))

    for ed in data.get('condominiums', []):
        address = safe_json_parse(ed.get('address'))
        neigh = address.get('neighborhood', 'Não informado')
        notas = ' '.join(ed.get('historicalNotes', []))
        
        content = (
            f"SOBRE O EDIFÍCIO {ed.get('name')} (em {ed.get('city')}, bairro {neigh}): "
            f"Status atual é {ed.get('buildingStatus', 'Não informado')}. "
            f"Notas ou história: {notas if notas else 'Sem notas históricas.'}"
        )
        docs.append(Document(page_content=sanitize_text(content)))

    for co in data.get('constructors', []):
        content = f"SOBRE A CONSTRUTORA {co.get('name')} (atua em {co.get('city')}): Descrição: {co.get('description', 'Não informada')}."
        docs.append(Document(page_content=sanitize_text(content)))

    return docs

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🏛️  VERTICALIZA LUME - ULTRA OTIMIZADO")
    print("="*60)
    
    print("📥 Extraindo dados da base do Verticaliza...")
    raw_data = fetch_hygraph_data()
    
    if not raw_data:
        print("🛑 Falha crítica ao carregar dados.")
        exit(1)
        
    print("⚡ Construindo Cérebro Local (Agregados calculados na memória)...")
    local_cache = VerticalizaCache(raw_data)
    
    cidades_nomes = [c.get('nome') for c in local_cache.cities.values()]
    cidades_str = ', '.join(cidades_nomes) if cidades_nomes else 'Nenhuma'
    
    print("🧩 Preparando motor semântico (FAISS) com dados curtos...")
    documentos = build_documents(raw_data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(documentos, embeddings)
    
    print("🤖 Inicializando Inteligência Artificial (Groq)...")
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        max_retries=0,
        timeout=25
    )

    print("🟢 LUME Online. (digite 'sair' para encerrar).\n")

    # Histórico reduzido para apenas 1 interação (Salva muita cota do limite de 6000 TPM do Groq)
    historico_chat = []

    while True:
        try:
            pergunta = input("\n👤 VOCÊ: ")
        except EOFError:
            break
            
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("✨ LUME: As luzes da cidade se apagam. Até logo.")
            break
            
        if not pergunta.strip():
            continue
            
        print("🤖 LUME analisando...")
        
        try:
            # 1. PROCESSAMENTO LOCAL
            dados_locais = local_cache.analyze_query(pergunta)
            
            # 2. BUSCA SEMÂNTICA (K=3 para poupar tokens drásticamente e focar na precisão)
            documentos_relevantes = vectorstore.similarity_search(pergunta, k=3)
            contexto_semantico = "\n".join([doc.page_content for doc in documentos_relevantes])
            
            texto_historico = ""
            for item in historico_chat:
                texto_historico += f"Usuário: {item['pergunta']}\nLUME: {item['resposta'][:100]}...\n"
            
            # 3. PROMPT COMPACTO E EXTREMAMENTE RESTRITO
            prompt_sistema = (
                f"Você é o LUME. O Verticaliza atua APENAS nestas cidades: {cidades_str}.\n\n"
                "REGRAS ABSOLUTAS:\n"
                "1. Use APENAS as informações de [DADOS LOCAIS] e [BUSCA SEMÂNTICA] abaixo.\n"
                "2. É ESTRITAMENTE PROIBIDO usar conhecimentos externos sobre geografia, montanhas, história ou economia.\n"
                "3. Se a informação não estiver nos blocos abaixo, diga: 'Não encontrei registros sobre isso na base.'\n"
                "4. Seja direto, não se apresente novamente.\n\n"
                f"[HISTÓRICO]:\n{texto_historico}\n"
                f"[DADOS LOCAIS]:\n{dados_locais}\n"
                f"[BUSCA SEMÂNTICA]:\n{contexto_semantico}"
            )
            
            mensagens = [
                ("system", prompt_sistema),
                ("human", pergunta)
            ]
            
            resposta = llm.invoke(mensagens)
            resposta_texto = resposta.content
            
            print(f"\n✨ LUME:\n{resposta_texto}")
            
            historico_chat = [{
                "pergunta": pergunta,
                "resposta": resposta_texto
            }]
            
        except Exception as e:
            if "429" in str(e) or "rate_limit_exceeded" in str(e):
                print("\n❌ LUME: Você fez perguntas muito rápido e atingiu o limite gratuito do Groq (6000 TPM). Aguarde 10 segundos e mande novamente.")
            else:
                print(f"\n❌ Erro: {e}")