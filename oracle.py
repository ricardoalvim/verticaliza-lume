import os
import sys
import json
import requests
import re
import time
import unicodedata
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuração da UI (deve ser o primeiro comando do Streamlit)
st.set_page_config(page_title="Verticaliza LUME", page_icon="🏛️", layout="centered")

# Força o terminal a lidar com caracteres estranhos (útil para logs no backend)
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
    clean_text = re.sub(r'\x1b\[.*?m', '', clean_text)
    return clean_text.encode('utf-8', 'ignore').decode('utf-8')

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

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

def format_large_number(num):
    """Formata números grandes para leitura fácil (ex: bilhões, milhões, milhares)."""
    try:
        if num is None or str(num).strip() == "":
            return "Não informado"
        n = float(num)
        if n >= 1_000_000_000:
            return f"{n/1_000_000_000:.2f} bilhões".replace('.', ',')
        elif n >= 1_000_000:
            return f"{n/1_000_000:.2f} milhões".replace('.', ',')
        elif n >= 1_000:
            return f"{n:,.0f}".replace(',', '.')
        return str(int(n))
    except:
        return str(num)

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
        coverImage { url }
        currentImages { url }
        historicalImages { url }
        floorPlanImages { url }
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
# 🧠 CÉREBRO LOCAL (IN-MEMORY CACHE E BUSCA EXATA)
# =====================================================================
class VerticalizaCache:
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
            timeline = safe_json_parse(ed.get('timeline'))
            
            raw_constructors = team.get('constructors', [])
            cons_names = [c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in raw_constructors]
            
            cover = ed.get('coverImage') or {}
            current = ed.get('currentImages') or []
            historical = ed.get('historicalImages') or []
            floor_plans = ed.get('floorPlanImages') or []
            
            # Formata o arquiteto de forma mais robusta
            arq_data = team.get('architect', {})
            arquiteto_nome = arq_data.get('name') if isinstance(arq_data, dict) else arq_data
            
            self.buildings[ed.get('name')] = {
                "nome": ed.get('name'),
                "cidade": ed.get('city'),
                "status": ed.get('buildingStatus', 'Não informado'),
                "segmento": ed.get('segment', 'Não informado'),
                "tipo": ed.get('type', 'Não informado'),
                "andares": specs.get('floors', 0),
                "quartos": specs.get('bedrooms', 0),
                "area": specs.get('unitArea', 0),
                "construtoras": cons_names,
                "arquiteto": arquiteto_nome if arquiteto_nome else 'Não informado',
                "timeline": timeline,
                "cover_url": cover.get('url') if cover else "",
                "current_urls": [img.get('url') for img in current if img.get('url')],
                "historical_urls": [img.get('url') for img in historical if img.get('url')],
                "floor_plan_urls": [img.get('url') for img in floor_plans if img.get('url')]
            }

        for c in data.get('cities', []):
            self.cities[c.get('name')] = {
                "nome": c.get('name'),
                "populacao": c.get('population'),
                "pib": c.get('gdp'),
                "infraestrutura": c.get('infrastructure', ''),
                "economia": c.get('economy', ''),
                "cultura": c.get('culture', ''),
                "turismo": c.get('tourism', ''),
                "total_edificios": 0 
            }

        for co in data.get('constructors', []):
            self.constructors[co.get('name')] = {
                "nome": co.get('name'),
                "fundacao": co.get('foundedYear', 'Não informado'),
                "total_obras": 0, 
                "obras": []
            }

    def _calculate_aggregates(self):
        for b_key, b_data in self.buildings.items():
            cidade_nome = str(b_data['cidade']).lower()
            for c_key, c_val in self.cities.items():
                if c_val['nome'].lower() == cidade_nome:
                    self.cities[c_key]['total_edificios'] += 1
            
            for c_name in b_data['construtoras']:
                c_name_lower = str(c_name).lower()
                for co_key, co_val in self.constructors.items():
                    if co_val['nome'].lower() == c_name_lower:
                        self.constructors[co_key]['total_obras'] += 1
                        self.constructors[co_key]['obras'].append(b_data['nome'])

    def analyze_query(self, query):
        q_norm = remove_accents(query.lower())
        contexto_local = []
        
        # Extrai a quantidade desejada pelo usuário (ex: "os 10 maiores", "top 3")
        qtd_match = re.search(r'\b(\d+)\b', q_norm)
        qtd_desejada = int(qtd_match.group(1)) if qtd_match else 5

        # 1. Checa Menções a Cidades
        cidade_mencionada = None
        for c_key, c_data in self.cities.items():
            if remove_accents(c_data['nome'].lower()) in q_norm:
                cidade_mencionada = c_key
                break

        if cidade_mencionada:
            c_data = self.cities[cidade_mencionada]
            
            # Textos ricos para infraestrutura e afins
            infra_str = f" Infraestrutura: {c_data['infraestrutura']}." if c_data['infraestrutura'] else " Infraestrutura: Sem dados catalogados."
            econ_str = f" Economia: {c_data['economia']}." if c_data['economia'] else " Economia: Sem dados catalogados."
            cult_str = f" Cultura/Eventos: {c_data['cultura']}." if c_data['cultura'] else " Cultura: Sem dados catalogados."
            
            contexto_local.append(
                f"DADOS GERAIS DA CIDADE {c_data['nome'].upper()}:\n"
                f"- População: {format_large_number(c_data['populacao'])}\n"
                f"- PIB: R$ {format_large_number(c_data['pib'])}\n"
                f"- Edifícios monitorados: {c_data['total_edificios']}\n"
                f"- Infraestrutura (escolas, universidades, hospitais, internet): {infra_str}\n"
                f"- Economia: {econ_str}\n"
                f"- Cultura e Eventos: {cult_str}\n"
                f"- Turismo: {c_data['turismo'] or 'Sem dados catalogados'}"
            )

            # === PANORAMA IMOBILIÁRIO AUTOMÁTICO ===
            predios_cidade = [b for b in self.buildings.values() if str(b['cidade']).lower() == str(self.cities[cidade_mencionada]['nome']).lower()]
            
            if predios_cidade:
                def pega_andares(x):
                    try: return int(x['andares'])
                    except: return 0
                
                def pega_ano(x):
                    t = x.get('timeline', {})
                    textos = [str(t.get('completion', '')), str(t.get('constructionStart', '')), str(t.get('announced', ''))]
                    for texto in textos:
                        anos = re.findall(r'\b(19\d{2}|20\d{2})\b', texto)
                        if anos: return int(anos[0])
                    return 9999 

                is_especifica = any(word in q_norm for word in ["maiores", "altos", "antigo", "velho", "primeiro", "obras", "construcao"])

                if not is_especifica:
                    concluidos = sum(1 for p in predios_cidade if str(p['status']).lower() in ['completed', 'concluído', 'pronto'])
                    em_obras = sum(1 for p in predios_cidade if str(p['status']).lower() in ['construction', 'em obras', 'under_construction', 'planned'])
                    
                    construtoras_cidade = {}
                    arquitetos_cidade = {}
                    
                    for p in predios_cidade:
                        for c in p['construtoras']:
                            if c and c.lower() not in ['não informado', 'não informada', 'n/a', '']:
                                construtoras_cidade[c] = construtoras_cidade.get(c, 0) + 1
                        
                        arq = p['arquiteto']
                        if arq and arq.lower() not in ['não informado', 'não informada', 'n/a', '']:
                            arquitetos_cidade[arq] = arquitetos_cidade.get(arq, 0) + 1

                    top_construtoras = sorted(construtoras_cidade.items(), key=lambda x: x[1], reverse=True)[:3]
                    top_arquitetos = sorted(arquitetos_cidade.items(), key=lambda x: x[1], reverse=True)[:3]

                    top_const_str = ", ".join([f"{c[0]} ({c[1]} obras)" for c in top_construtoras]) if top_construtoras else "Ainda não catalogadas"
                    top_arq_str = ", ".join([f"{a[0]} ({a[1]} projetos)" for a in top_arquitetos]) if top_arquitetos else "Ainda não catalogados"

                    mais_alto = sorted(predios_cidade, key=pega_andares, reverse=True)[0] if predios_cidade else None
                    mais_antigo = sorted(predios_cidade, key=pega_ano)[0] if predios_cidade else None

                    resumo = (
                        f"\nPANORAMA IMOBILIÁRIO DE {c_data['nome'].upper()}:\n"
                        f"- Total de edifícios monitorados: {len(predios_cidade)} ({concluidos} concluídos, {em_obras} em projeto/obras).\n"
                        f"- Construtoras com mais presença na base: {top_const_str}.\n"
                        f"- Arquitetos mais catalogados: {top_arq_str}.\n"
                    )
                    
                    if mais_alto and pega_andares(mais_alto) > 0:
                        img_alto = f" ![Fachada]({mais_alto['cover_url']})" if mais_alto['cover_url'] else ""
                        resumo += f"- Edifício mais alto: {mais_alto['nome']} ({format_value(mais_alto['andares'])} andares | Segmento: {mais_alto['segmento']} | Quartos: {format_value(mais_alto['quartos'])} | Construtora: {', '.join(mais_alto['construtoras']) if mais_alto['construtoras'] else 'Não informada'} | Status: {mais_alto['status']}).{img_alto}\n"
                    
                    if mais_antigo:
                        ano_antigo = pega_ano(mais_antigo)
                        if ano_antigo != 9999:
                            img_antigo = f" ![Fachada]({mais_antigo['cover_url']})" if mais_antigo['cover_url'] else ""
                            resumo += f"- Edifício mais antigo registrado: {mais_antigo['nome']} (Ano referência: {ano_antigo} | Segmento: {mais_antigo['segmento']} | Construtora: {', '.join(mais_antigo['construtoras']) if mais_antigo['construtoras'] else 'Não informada'} | Status: {mais_antigo['status']}).{img_antigo}\n"

                    contexto_local.append(resumo)

                if "maiores" in q_norm or "mais altos" in q_norm:
                    predios_ordenados = sorted(predios_cidade, key=pega_andares, reverse=True)[:qtd_desejada]
                    contexto_local.append(f"\nOS {len(predios_ordenados)} MAIORES EDIFÍCIOS EM {self.cities[cidade_mencionada]['nome'].upper()}:")
                    for i, p in enumerate(predios_ordenados, 1):
                        contexto_local.append(
                            f"{i}. {p['nome']} - {format_value(p['andares'])} andares | "
                            f"Segmento: {p['segmento']} | Quartos: {format_value(p['quartos'])} | Área: {format_value(p['area'], 'm²')} | "
                            f"Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'} | Status: {p['status']}"
                        )
                        
                if "antigo" in q_norm or "velho" in q_norm or "primeiro" in q_norm:
                    predios_ordenados = sorted(predios_cidade, key=pega_ano)[:qtd_desejada]
                    contexto_local.append(f"\nOS {len(predios_ordenados)} EDIFÍCIOS MAIS ANTIGOS EM {self.cities[cidade_mencionada]['nome'].upper()}:")
                    for i, p in enumerate(predios_ordenados, 1):
                        ano = pega_ano(p)
                        ano_str = ano if ano != 9999 else "Data desconhecida"
                        contexto_local.append(
                            f"{i}. {p['nome']} - Ano/Referência: {ano_str} | {format_value(p['andares'])} andares | "
                            f"Segmento: {p['segmento']} | Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'} | "
                            f"Status: {p['status']}"
                        )

                if "obras" in q_norm or "construcao" in q_norm:
                    status_obras = ['construção', 'em obras', 'under_construction', 'construction']
                    predios_obras = [b for b in predios_cidade if str(b['status']).lower() in status_obras]
                    
                    if predios_obras:
                        contexto_local.append(f"\nEDIFÍCIOS EM CONSTRUÇÃO EM {self.cities[cidade_mencionada]['nome'].upper()}:")
                        for p in predios_obras:
                            contexto_local.append(
                                f"- {p['nome']} ({format_value(p['andares'])} andares | "
                                f"Segmento: {p['segmento']} | Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'})"
                            )
                    else:
                        contexto_local.append(f"\nNão há edifícios em construção registrados em {self.cities[cidade_mencionada]['nome']}.")

        # 2. Checa Menções a Construtoras e Top N da Construtora
        for c_key, c_data in self.constructors.items():
            if remove_accents(c_data['nome'].lower()) in q_norm:
                contexto_local.append(
                    f"\nDADOS DA CONSTRUTORA '{c_data['nome']}':\n"
                    f"- Obras registradas na base: {c_data['total_obras']}\n"
                    f"- Principais Obras: {', '.join(c_data['obras'][:10])}."
                )
                
                if "maiores" in q_norm or "mais altos" in q_norm:
                    predios_const = [self.buildings[nome] for nome in c_data['obras'] if nome in self.buildings]
                    def pega_andares_c(x):
                        try: return int(x['andares'])
                        except: return 0
                    
                    predios_ord = sorted(predios_const, key=pega_andares_c, reverse=True)[:qtd_desejada]
                    if predios_ord:
                        contexto_local.append(f"\nOS {len(predios_ord)} MAIORES EDIFÍCIOS DA CONSTRUTORA {c_data['nome'].upper()}:")
                        for i, p in enumerate(predios_ord, 1):
                            contexto_local.append(f"{i}. {p['nome']} - {format_value(p['andares'])} andares (Status: {p['status']})")
                            
                if "antigo" in q_norm or "velho" in q_norm or "primeiro" in q_norm:
                    predios_const = [self.buildings[nome] for nome in c_data['obras'] if nome in self.buildings]
                    def pega_ano_c(x):
                        t = x.get('timeline', {})
                        textos = [str(t.get('completion', '')), str(t.get('constructionStart', '')), str(t.get('announced', ''))]
                        for texto in textos:
                            anos = re.findall(r'\b(19\d{2}|20\d{2})\b', texto)
                            if anos: return int(anos[0])
                        return 9999 
                    
                    predios_ord = sorted(predios_const, key=pega_ano_c)[:qtd_desejada]
                    if predios_ord:
                        contexto_local.append(f"\nOS {len(predios_ord)} EDIFÍCIOS MAIS ANTIGOS DA CONSTRUTORA {c_data['nome'].upper()}:")
                        for i, p in enumerate(predios_ord, 1):
                            ano = pega_ano_c(p)
                            ano_str = ano if ano != 9999 else "Data desconhecida"
                            contexto_local.append(f"{i}. {p['nome']} - Ano: {ano_str} (Status: {p['status']})")

        # 3. Checa Menções a Edifícios Específicos
        mencoes_predios = []
        q_flex = re.sub(r'(.)\1+', r'\1', q_norm)
        
        for b_key, b_data in self.buildings.items():
            nome_completo = b_data['nome']
            nome_core = re.sub(r'^(edif[íi]cio|residencial|condom[íi]nio|hotel|torre|complexo|ed\.|res\.)\s+', '', nome_completo, flags=re.IGNORECASE).strip()
            
            n_comp_norm = remove_accents(nome_completo.lower())
            n_core_norm = remove_accents(nome_core.lower())
            n_core_flex = re.sub(r'(.)\1+', r'\1', n_core_norm)
            
            if len(n_core_flex) >= 3 and (n_core_flex in q_flex or n_core_norm in q_norm or n_comp_norm in q_norm or (len(q_norm) > 4 and q_norm in n_core_norm)):
                if b_data not in mencoes_predios:
                    mencoes_predios.append(b_data)
        
        if ("compar" in q_norm or "diferenca" in q_norm) and len(mencoes_predios) >= 2:
            contexto_local.append("\nCOMPARAÇÃO DE EDIFÍCIOS:")
            for p in mencoes_predios:
                img_str = f" | ![Fachada]({p['cover_url']})" if p['cover_url'] else ""
                contexto_local.append(
                    f"- {p['nome']} ({p['cidade']}): {format_value(p['andares'])} andares, "
                    f"Segmento: {p['segmento']}, Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'}, Status: {p['status']}{img_str}"
                )
        elif len(mencoes_predios) > 0:
            for p in mencoes_predios:
                img_strs = []
                if p['cover_url']: img_strs.append(f"![Fachada do {p['nome']}]({p['cover_url']})")
                for i, img in enumerate(p['floor_plan_urls'][:2]): img_strs.append(f"![Planta {i+1}]({img})")
                for i, img in enumerate(p['historical_urls'][:2]): img_strs.append(f"![Imagem Histórica {i+1}]({img})")
                
                img_str_final = "\nIMAGENS (Copie os códigos Markdown caso o usuário queira ver fotos/plantas):\n" + "\n".join(img_strs) if img_strs else ""
                
                t_data = p.get('timeline', {})
                inicio_obras = format_value(t_data.get('constructionStart'))
                conclusao = format_value(t_data.get('completion'))
                
                contexto_local.append(
                    f"\nFICHA TÉCNICA DO EDIFÍCIO '{p['nome']}':\n"
                    f"- Localização: {p['cidade']}\n"
                    f"- Segmento/Tipo: {p['segmento']} / {p['tipo']}\n"
                    f"- Andares: {format_value(p['andares'])}\n"
                    f"- Quartos: {format_value(p['quartos'])}\n"
                    f"- Área: {format_value(p['area'], 'm²')}\n"
                    f"- Status Atual: {p['status']}\n"
                    f"- Construtora(s): {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'}\n"
                    f"- Arquiteto(s) Responsável: {p['arquiteto']}\n"
                    f"- Início das Obras: {inicio_obras}\n"
                    f"- Ano de Conclusão: {conclusao}\n"
                    f"{img_str_final}"
                )

        return "\n".join(contexto_local)

# =====================================================================
# 📚 MOTOR SEMÂNTICO (FAISS)
# =====================================================================
def build_documents(data):
    if not data:
        return []
    docs = []

    for c in data.get('cities', []):
        content = f"DADOS DA CIDADE {c.get('name')} ({c.get('state')}): Fica na região {c.get('region')}."
        docs.append(Document(page_content=sanitize_text(content)))

    for ed in data.get('condominiums', []):
        address = safe_json_parse(ed.get('address'))
        neigh = address.get('neighborhood', 'Não informado')
        notas = ' '.join(ed.get('historicalNotes', []))
        
        content = (
            f"EDIFÍCIO {ed.get('name')} em {ed.get('city')}, bairro {neigh}. "
            f"História e Notas: {notas if notas else 'Sem notas históricas.'}"
        )
        docs.append(Document(page_content=sanitize_text(content)))

    for co in data.get('constructors', []):
        content = f"CONSTRUTORA {co.get('name')} ({co.get('city')}): Descrição e História: {co.get('description', 'Não informada')}."
        docs.append(Document(page_content=sanitize_text(content)))

    return docs

# =====================================================================
# ⚡ INICIALIZAÇÃO DA IA NO STREAMLIT (Executado apenas 1x)
# =====================================================================
@st.cache_resource(show_spinner="Preparando o motor de busca e inteligência...")
def setup_ia_system():
    raw_data = fetch_hygraph_data()
    if not raw_data:
        return None, None, None, ""
        
    local_cache = VerticalizaCache(raw_data)
    cidades_nomes = [c.get('nome') for c in local_cache.cities.values()]
    cidades_str = ', '.join(cidades_nomes) if cidades_nomes else 'Nenhuma'
    
    documentos = build_documents(raw_data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(documentos, embeddings)
    
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        max_retries=0,
        timeout=25
    )
    
    return local_cache, vectorstore, llm, cidades_str

# =====================================================================
# 🖥️ INTERFACE WEB E LOOP PRINCIPAL DO CHAT
# =====================================================================
st.title("🏛️ Verticaliza LUME")

local_cache, vectorstore, llm, cidades_str = setup_ia_system()

if not local_cache:
    st.error("🛑 Falha crítica ao carregar dados da base do Verticaliza.")
    st.stop()

# Saudação Inicial Rica e Direta
boas_vindas_texto = """Olá! Eu sou o **LUME**, a Inteligência Artificial do projeto **Verticaliza**. 🏢✨

Sou especialista em urbanismo, infraestrutura, cultura local e no mercado imobiliário da região. 

Você pode me perguntar sobre:
- 🏙️ **Edifícios:** Fichas técnicas, construtoras, andares, arquitetos responsáveis e status.
- 🏗️ **Construtoras:** Histórico, portfólio e quantidade de obras.
- 🌆 **Cidades:** Economia, eventos culturais, turismo, hospitais, universidades e provedores de infraestrutura.

*I speak English. Hablo Español. Ich spreche Deutsch.*

Como posso ajudar na sua pesquisa hoje?"""

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": boas_vindas_texto}]

if "ultima_pergunta_usuario" not in st.session_state:
    st.session_state.ultima_pergunta_usuario = ""

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pergunta := st.chat_input("👤 VOCÊ:"):
    
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Aguarde..."):
            time.sleep(2.5)
            
            try:
                # Expansor Inteligente com mais contexto 
                query_expandida = pergunta
                palavras_contexto = ['ele', 'ela', 'dele', 'dela', 'desse', 'dessa', 'quando', 'onde', 'arquiteto', 'engenheiro', 'projetou', 'plantas', 'universidade', 'faculdade', 'cultura', 'e', 'mas', 'não', 'nao', 'qual', 'quais']
                if st.session_state.ultima_pergunta_usuario and (len(pergunta.split()) < 7 or any(p in pergunta.lower().split() for p in palavras_contexto)):
                    query_expandida = f"{st.session_state.ultima_pergunta_usuario} | {pergunta}"

                # 1. PROCESSAMENTO LOCAL
                dados_locais = local_cache.analyze_query(query_expandida)
                
                # 2. BUSCA SEMÂNTICA
                contexto_semantico = ""
                try:
                    documentos_relevantes = vectorstore.similarity_search(query_expandida, k=3)
                    contexto_semantico = "\n".join([doc.page_content for doc in documentos_relevantes])
                except Exception:
                    pass
                
                # 3. PROMPT BLINDADO E REESTRUTURADO (Fim das Tags XML)
                prompt_sistema = (
                    "Você é o LUME, a Inteligência Artificial especialista em urbanismo e mercado imobiliário do projeto Verticaliza.\n"
                    "Aja como um consultor humano especialista. Todo o conhecimento abaixo já faz parte da sua mente.\n\n"
                    "REGRAS DE OURO (PUNIÇÃO MÁXIMA SE DESCUMPRIDAS):\n"
                    "1. NUNCA MENCIONE SUAS FONTES: É ESTRITAMENTE PROIBIDO usar expressões como 'De acordo com os dados', 'Na tag', 'Meus registros', 'A base de dados diz' ou 'Fui fornecido'. Responda DIRETAMENTE como um especialista (Ex: Em vez de 'Os dados dizem que tem 5 andares', diga 'O prédio tem 5 andares').\n"
                    "2. ALUCINAÇÃO ZERO: Não invente cursos de universidades, estilos arquitetônicos, festas, características de prédios ou prazos que não estejam explicitamente listados no SEU CONHECIMENTO abaixo. Se não souber, diga apenas: 'Ainda não possuo essa informação catalogada.'\n"
                    "3. IDIOMA: Responda em Português do Brasil (proibido usar 'tu', 'fizestes', 'tens').\n"
                    "4. IMAGENS: Se o contexto fornecer um código Markdown de imagem (ex: ![Texto](url)), copie-o EXATAMENTE para a sua resposta. Não invente imagens ou crie links do Google.\n"
                    "5. RANKINGS E OBRAS: Se o usuário pedir maiores/antigos, cite EXATAMENTE as listas fornecidas abaixo, sem inventar ou suprimir informações listadas.\n\n"
                    "--- SEU CONHECIMENTO ---\n"
                    f"CIDADES DE ATUAÇÃO: {cidades_str}.\n"
                    f"{dados_locais if dados_locais else ''}\n"
                    f"{contexto_semantico if contexto_semantico else ''}\n"
                    "--------------------------"
                )
                
                mensagens = [SystemMessage(content=prompt_sistema)]
                
                for msg in st.session_state.messages[-5:]:
                    if msg["role"] == "user":
                        mensagens.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant" and not "🏢✨" in msg["content"]:
                        mensagens.append(AIMessage(content=msg["content"]))
                
                resposta = llm.invoke(mensagens)
                resposta_texto = resposta.content
                
                st.markdown(resposta_texto)
                st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
                
                st.session_state.ultima_pergunta_usuario = pergunta
                
            except Exception as e:
                if "429" in str(e) or "rate_limit_exceeded" in str(e):
                    st.error("❌ LUME: Limite do Groq atingido. Aguarde alguns segundos.")
                else:
                    st.error(f"❌ Erro interno: {e}")