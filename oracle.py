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

# Configuração da UI
st.set_page_config(page_title="Verticaliza LUME", page_icon="🏛️", layout="wide")

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
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
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

# === BUSCA APENAS AS CIDADES (RÁPIDO E LEVE) ===
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_base_cities():
    headers = {"Authorization": f"Bearer {HYGRAPH_TOKEN}"}
    query = """
    query {
      cities(stage: PUBLISHED, first: 1000) { 
        name slug state region population area gdp gdpPerCapita averageIncome medianIncome idh geoloc { latitude longitude } infrastructure culture tourism economy state
      }
    }
    """
    try:
        response = requests.post(HYGRAPH_URL, json={'query': query}, headers=headers)
        return response.json().get('data', {}).get('cities', [])
    except Exception as e:
        st.error(f"❌ Erro ao consultar as cidades base: {e}")
        return []

# === BUSCA TODAS AS ENTIDADES PARA FILTRAGEM SEGURA EM PYTHON ===
def fetch_all_entities():
    headers = {"Authorization": f"Bearer {HYGRAPH_TOKEN}"}
    query = """
    query {
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
        richContent { text }
      }
    }
    """
    try:
        response = requests.post(HYGRAPH_URL, json={'query': query}, headers=headers)
        res_json = response.json()
        if 'errors' in res_json:
            st.error(f"❌ Erro do Hygraph: {res_json['errors'][0].get('message')}")
            return {}
        return res_json.get('data', {})
    except Exception as e:
        st.error(f"❌ Erro de rede ao consultar entidades: {e}")
        return {}

# =====================================================================
# 🧠 CÉREBRO LOCAL (FOCADO EM APENAS 1 CIDADE)
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
        for c in data.get('cities', []):
            self.cities[c.get('name')] = {
                "nome": c.get('name'),
                "slug": c.get('slug', ''),
                "populacao": c.get('population'),
                "pib": c.get('gdp'),
                "idh": c.get('idh', 'Não catalogado'),
                "area": c.get('area', 'Não catalogada'),
                "infraestrutura": c.get('infrastructure', ''),
                "economia": c.get('economy', ''),
                "cultura": c.get('culture', ''),
                "turismo": c.get('tourism', ''),
                "total_edificios": len(data.get('condominiums', []))
            }

        for co in data.get('constructors', []):
            self.constructors[co.get('name')] = {
                "nome": co.get('name'),
                "fundacao": co.get('foundedYear', 'Não informado'),
                "total_obras": 0, 
                "obras": []
            }

        for ed in data.get('condominiums', []):
            specs = safe_json_parse(ed.get('specifications'))
            team = safe_json_parse(ed.get('team'))
            timeline = safe_json_parse(ed.get('timeline'))
            address = safe_json_parse(ed.get('address'))
            seo = safe_json_parse(ed.get('seo'))
            
            raw_constructors = team.get('constructors', [])
            cons_names = [c.get('name', str(c)) if isinstance(c, dict) else str(c) for c in raw_constructors]
            
            cover = ed.get('coverImage') or {}
            current = ed.get('currentImages') or []
            historical = ed.get('historicalImages') or []
            floor_plans = ed.get('floorPlanImages') or []
            
            rich_content = ed.get('richContent') or {}
            historia_rica = rich_content.get('text', '') if isinstance(rich_content, dict) else ""
            notas_historicas = ed.get('historicalNotes', [])
            notas_str = " ".join(notas_historicas) if notas_historicas else ""
            
            rua = address.get('street', '')
            num = address.get('number', '')
            bairro = address.get('neighborhood', '')
            endereco_completo = f"{rua}, {num} - Bairro {bairro}".strip(" ,-")
            
            arq_data = team.get('architect', {})
            arquiteto_nome = arq_data.get('name') if isinstance(arq_data, dict) else arq_data
            eng_estrutural = team.get('structuralEngineer', {}).get('name', '') if isinstance(team.get('structuralEngineer'), dict) else ''
            
            nome_ed = ed.get('name')
            self.buildings[nome_ed] = {
                "nome": nome_ed,
                "cidade": ed.get('city'),
                "status": ed.get('buildingStatus', 'Não informado'),
                "segmento": ed.get('segment', 'Não informado'),
                "tipo": ed.get('type', 'Não informado'),
                "endereco": endereco_completo,
                "andares": specs.get('floors', 0),
                "quartos": specs.get('bedrooms', 0),
                "area": specs.get('unitArea', 0),
                "construtoras": cons_names,
                "arquiteto": arquiteto_nome if arquiteto_nome else 'Não informado',
                "eng_estrutural": eng_estrutural,
                "timeline": timeline,
                "historia": historia_rica,
                "notas": notas_str,
                "seo_desc": seo.get('description', ''),
                "cover_url": cover.get('url') if cover else "",
                "current_urls": [img.get('url') for img in current if img.get('url')],
                "historical_urls": [img.get('url') for img in historical if img.get('url')],
                "floor_plan_urls": [img.get('url') for img in floor_plans if img.get('url')]
            }

    def _calculate_aggregates(self):
        cidade_ativa = list(self.cities.keys())[0] if self.cities else None
        
        for b_key, b_data in self.buildings.items():
            if cidade_ativa:
                self.cities[cidade_ativa]['total_edificios'] = len(self.buildings)
            
            for c_name in b_data['construtoras']:
                c_name_lower = remove_accents(str(c_name).lower().strip())
                for co_key, co_val in self.constructors.items():
                    if remove_accents(co_val['nome'].lower().strip()) == c_name_lower:
                        self.constructors[co_key]['total_obras'] += 1
                        self.constructors[co_key]['obras'].append(b_data['nome'])

    def analyze_query(self, query):
        q_norm = remove_accents(query.lower())
        contexto_local = []
        
        # 🚨 GUARD RAIL 1: KILL SWITCH PARA SÍNTESE HISTÓRICA/ECONÔMICA
        palavras_proibidas = ["decada", "decadas", "seculo", "plano", "planos", "economico", "economicos", "cruze", "governo", "presidente", "politica", "inflacao"]
        if any(word in q_norm for word in palavras_proibidas):
            contexto_local.append("\n[BLOQUEIO DE SEGURANÇA: O Verticaliza não realiza cruzamentos com planos econômicos ou análises históricas por década. Responda que o escopo é exclusivamente técnico.]")
        
        qtd_match = re.search(r'\b(\d+)\b', q_norm)
        qtd_desejada = int(qtd_match.group(1)) if qtd_match else 5

        is_maiores = any(word in q_norm for word in ["maior", "alto", "top", "tall", "high", "big", "mayor", "hoch", "gros"])
        is_antigo = any(word in q_norm for word in ["antigo", "velho", "primeiro", "old", "first", "antiguo", "primer", "alt", "erste"])
        is_recente = any(word in q_norm for word in ["recente", "novo", "ultima", "ultimo", "recent", "new", "latest", "reciente", "nuevo", "neu", "letzte"])
        is_obras = any(word in q_norm for word in ["obra", "construcao", "construction", "building", "construccion", "baustelle", "bau", "andamento"])
        is_especifica = is_maiores or is_antigo or is_recente or is_obras

        somente_concluidos = any(word in q_norm for word in ["concluido", "concluidos", "pronto", "prontos", "entregue", "entregues", "finalizado"])

        c_data = list(self.cities.values())[0]
            
        # 🚨 GUARD RAIL 2: TRAVAS DIRETAS NO DOSSIÊ DA CIDADE
        infra_str = f" Infraestrutura/Universidades: {c_data['infraestrutura']}" if c_data['infraestrutura'] else " Infraestrutura: [FONTE CMS: DADO NÃO CATALOGADO - PROIBIDO INVENTAR]"
        econ_str = f" Economia: {c_data['economia']}" if c_data['economia'] else " Economia: [FONTE CMS: DADO NÃO CATALOGADO - PROIBIDO INVENTAR]"
        cult_str = f" Cultura/Eventos: {c_data['cultura']}" if c_data['cultura'] else " Cultura: [FONTE CMS: DADO NÃO CATALOGADO - PROIBIDO INVENTAR]"
        tur_str = f" Turismo: {c_data['turismo']}" if c_data['turismo'] else " Turismo: [FONTE CMS: DADO NÃO CATALOGADO - PROIBIDO INVENTAR]"
        
        contexto_local.append(
            f"DADOS GERAIS DA CIDADE ATUAL ({c_data['nome'].upper()}):\n"
            f"- População: {format_large_number(c_data['populacao'])}\n"
            f"- PIB: R$ {format_large_number(c_data['pib'])}\n"
            f"- IDH: {c_data['idh']}\n"
            f"- Área: {c_data['area']} km²\n"
            f"- Edifícios monitorados no Verticaliza: {c_data['total_edificios']}\n"
            f"- {infra_str}\n"
            f"- {econ_str}\n"
            f"- {cult_str}\n"
            f"- {tur_str}"
        )

        predios_cidade = list(self.buildings.values())
        predios_para_ranking = [p for p in predios_cidade if str(p['status']).lower() in ['completed', 'concluído', 'pronto']] if somente_concluidos else predios_cidade

        if not predios_cidade:
            contexto_local.append(f"\n[ALERTA CRÍTICO: ESTA CIDADE ESTÁ VAZIA NO CMS. NUNCA INVENTE EDIFÍCIOS PARA ELA.]")
        else:
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
            
            def pega_ano_desc(x):
                t = x.get('timeline', {})
                textos = [str(t.get('completion', '')), str(t.get('constructionStart', '')), str(t.get('announced', ''))]
                for texto in textos:
                    anos = re.findall(r'\b(19\d{2}|20\d{2})\b', texto)
                    if anos: return int(anos[0])
                return 0

            # Gerador de Dossiê Panorama
            if not is_especifica and ("edificio" in q_norm or "predio" in q_norm or remove_accents(c_data['nome'].lower()) in q_norm):
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

                resumo = (
                    f"\nPANORAMA IMOBILIÁRIO DE {c_data['nome'].upper()}:\n"
                    f"- Total de edifícios monitorados: {len(predios_cidade)} ({concluidos} concluídos, {em_obras} em projeto/obras).\n"
                    f"- Construtoras com mais presença na base: {top_const_str}.\n"
                    f"- Arquitetos mais catalogados: {top_arq_str}.\n"
                )
                
                mais_alto = sorted(predios_cidade, key=pega_andares, reverse=True)[0] if predios_cidade else None
                mais_antigo = sorted(predios_cidade, key=pega_ano)[0] if predios_cidade else None

                if mais_alto and pega_andares(mais_alto) > 0:
                    img_alto = f" ![Fachada]({mais_alto['cover_url']})" if mais_alto['cover_url'] else ""
                    resumo += f"- Edifício mais alto: {mais_alto['nome']} ({format_value(mais_alto['andares'])} andares | Segmento: {mais_alto['segmento']} | Quartos: {format_value(mais_alto['quartos'])} | Construtora: {', '.join(mais_alto['construtoras']) if mais_alto['construtoras'] else 'Não informada'} | Status: {mais_alto['status']}).{img_alto}\n"
                
                if mais_antigo:
                    ano_antigo = pega_ano(mais_antigo)
                    if ano_antigo != 9999:
                        img_antigo = f" ![Fachada]({mais_antigo['cover_url']})" if mais_antigo['cover_url'] else ""
                        resumo += f"- Edifício mais antigo registrado: {mais_antigo['nome']} (Ano referência: {ano_antigo} | Segmento: {mais_antigo['segmento']} | Construtora: {', '.join(mais_antigo['construtoras']) if mais_antigo['construtoras'] else 'Não informada'} | Status: {mais_antigo['status']}).{img_antigo}\n"

                contexto_local.append(resumo)

            if is_maiores and not any(remove_accents(c.lower()) in q_norm for c in self.constructors):
                predios_ordenados = sorted(predios_para_ranking, key=pega_andares, reverse=True)[:qtd_desejada]
                contexto_local.append(f"\nOS {len(predios_ordenados)} MAIORES EDIFÍCIOS EM {c_data['nome'].upper()}:")
                for i, p in enumerate(predios_ordenados, 1):
                    contexto_local.append(f"{i}. {p['nome']} - {format_value(p['andares'])} andares | Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'} | Status: {p['status']}")
                    
            if is_antigo and not any(remove_accents(c.lower()) in q_norm for c in self.constructors):
                predios_ordenados = sorted(predios_para_ranking, key=pega_ano)[:qtd_desejada]
                contexto_local.append(f"\nOS {len(predios_ordenados)} EDIFÍCIOS MAIS ANTIGOS EM {c_data['nome'].upper()}:")
                for i, p in enumerate(predios_ordenados, 1):
                    ano = pega_ano(p)
                    ano_str = ano if ano != 9999 else "Data desconhecida"
                    contexto_local.append(f"{i}. {p['nome']} - Ano/Referência: {ano_str} | {format_value(p['andares'])} andares | Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'} | Status: {p['status']}")

            if is_recente and not any(remove_accents(c.lower()) in q_norm for c in self.constructors):
                predios_ordenados = sorted(predios_para_ranking, key=pega_ano_desc, reverse=True)[:qtd_desejada]
                contexto_local.append(f"\nOS {len(predios_ordenados)} EDIFÍCIOS MAIS RECENTES EM {c_data['nome'].upper()}:")
                for i, p in enumerate(predios_ordenados, 1):
                    ano = pega_ano_desc(p)
                    ano_str = ano if ano != 0 else "Data desconhecida"
                    contexto_local.append(f"{i}. {p['nome']} - Ano/Referência: {ano_str} | {format_value(p['andares'])} andares | Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'} | Status: {p['status']}")

            if is_obras and not any(remove_accents(c.lower()) in q_norm for c in self.constructors):
                status_obras = ['construção', 'em obras', 'under_construction', 'construction', 'planned']
                predios_obras = [b for b in predios_cidade if str(b['status']).lower() in status_obras]
                if predios_obras:
                    contexto_local.append(f"\nEDIFÍCIOS EM CONSTRUÇÃO EM {c_data['nome'].upper()}:")
                    for p in predios_obras:
                        contexto_local.append(f"- {p['nome']} ({format_value(p['andares'])} andares | Construtora: {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'})")
                else:
                    contexto_local.append(f"\nNão há edifícios em construção registrados em {c_data['nome'].upper()}.")

        # 2. Checa Menções a Construtoras e Top N da Construtora
        for c_key, c_data in self.constructors.items():
            if remove_accents(c_data['nome'].lower()) in q_norm:
                if c_data['total_obras'] > 0:
                    # Injeta Mini-Dossiês para as obras
                    obras_detalhes = []
                    for nome_obra in c_data['obras'][:10]:
                        if nome_obra in self.buildings:
                            b = self.buildings[nome_obra]
                            obras_detalhes.append(f"{b['nome']} (Status: {b['status']} | {format_value(b['andares'])} andares)")
                        else:
                            obras_detalhes.append(nome_obra)

                    contexto_local.append(
                        f"\nDADOS DA CONSTRUTORA '{c_data['nome']}':\n"
                        f"- Obras registradas na base local: {c_data['total_obras']}\n"
                        f"- Resumo das Principais Obras: {', '.join(obras_detalhes)}."
                    )
                
                if is_maiores:
                    predios_const = [self.buildings[nome] for nome in c_data['obras'] if nome in self.buildings]
                    predios_const_filtrados = [p for p in predios_const if str(p['status']).lower() in ['completed', 'concluído', 'pronto']] if somente_concluidos else predios_const
                    
                    def pega_andares_c(x):
                        try: return int(x['andares'])
                        except: return 0
                    
                    predios_ord = sorted(predios_const_filtrados, key=pega_andares_c, reverse=True)[:qtd_desejada]
                    if predios_ord:
                        contexto_local.append(f"\nOS {len(predios_ord)} MAIORES EDIFÍCIOS DA CONSTRUTORA {c_data['nome'].upper()}:")
                        for i, p in enumerate(predios_ord, 1):
                            contexto_local.append(f"{i}. {p['nome']} - {format_value(p['andares'])} andares (Status: {p['status']})")
                            
                if is_antigo:
                    predios_const = [self.buildings[nome] for nome in c_data['obras'] if nome in self.buildings]
                    predios_const_filtrados = [p for p in predios_const if str(p['status']).lower() in ['completed', 'concluído', 'pronto']] if somente_concluidos else predios_const
                    
                    def pega_ano_c(x):
                        t = x.get('timeline', {})
                        textos = [str(t.get('completion', '')), str(t.get('constructionStart', '')), str(t.get('announced', ''))]
                        for texto in textos:
                            anos = re.findall(r'\b(19\d{2}|20\d{2})\b', texto)
                            if anos: return int(anos[0])
                        return 9999 
                    
                    predios_ord = sorted(predios_const_filtrados, key=pega_ano_c)[:qtd_desejada]
                    if predios_ord:
                        contexto_local.append(f"\nOS {len(predios_ord)} EDIFÍCIOS MAIS ANTIGOS DA CONSTRUTORA {c_data['nome'].upper()}:")
                        for i, p in enumerate(predios_ord, 1):
                            ano = pega_ano_c(p)
                            ano_str = ano if ano != 9999 else "Data desconhecida"
                            contexto_local.append(f"{i}. {p['nome']} - Ano: {ano_str} (Status: {p['status']})")

                if is_recente:
                    predios_const = [self.buildings[nome] for nome in c_data['obras'] if nome in self.buildings]
                    predios_const_filtrados = [p for p in predios_const if str(p['status']).lower() in ['completed', 'concluído', 'pronto']] if somente_concluidos else predios_const
                    
                    def pega_ano_c_desc(x):
                        t = x.get('timeline', {})
                        textos = [str(t.get('completion', '')), str(t.get('constructionStart', '')), str(t.get('announced', ''))]
                        for texto in textos:
                            anos = re.findall(r'\b(19\d{2}|20\d{2})\b', texto)
                            if anos: return int(anos[0])
                        return 0 
                    
                    predios_ord = sorted(predios_const_filtrados, key=pega_ano_c_desc, reverse=True)[:qtd_desejada]
                    if predios_ord:
                        contexto_local.append(f"\nAS {len(predios_ord)} OBRAS MAIS RECENTES DA CONSTRUTORA {c_data['nome'].upper()}:")
                        for i, p in enumerate(predios_ord, 1):
                            ano = pega_ano_c_desc(p)
                            ano_str = ano if ano != 0 else "Data desconhecida"
                            contexto_local.append(f"{i}. {p['nome']} - Ano: {ano_str} (Status: {p['status']})")

        # 3. Checa Menções a Edifícios Específicos (INJETANDO HISTÓRIA E ENDEREÇO)
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
                
                img_str_final = "\nIMAGENS:\n" + "\n".join(img_strs) if img_strs else ""
                
                t_data = p.get('timeline', {})
                inicio_obras = format_value(t_data.get('constructionStart'))
                conclusao = format_value(t_data.get('completion'))
                
                contexto_local.append(
                    f"\nDOSSIÊ COMPLETO DO EDIFÍCIO '{p['nome']}':\n"
                    f"- Localização: {p['cidade']}\n"
                    f"- Endereço Exato: {p['endereco'] if p['endereco'] else 'Rua não informada'}\n"
                    f"- Segmento/Tipo: {p['segmento']} / {p['tipo']}\n"
                    f"- Andares: {format_value(p['andares'])}\n"
                    f"- Quartos: {format_value(p['quartos'])}\n"
                    f"- Área: {format_value(p['area'], 'm²')}\n"
                    f"- Status Atual: {p['status']}\n"
                    f"- Construtora(s): {', '.join(p['construtoras']) if p['construtoras'] else 'Não informada'}\n"
                    f"- Arquiteto(s): {p['arquiteto']}\n"
                    f"- Engenheiro Estrutural: {p['eng_estrutural'] if p['eng_estrutural'] else 'Não informado'}\n"
                    f"- Início das Obras: {inicio_obras}\n"
                    f"- Ano de Conclusão: {conclusao}\n"
                    f"- Descrição SEO: {p['seo_desc']}\n"
                    f"- Notas Básicas: {p['notas'] if p['notas'] else 'Nenhuma nota registrada.'}\n"
                    f"- HISTÓRIA RICA (Artigos/Jornais): {p['historia'] if p['historia'] else 'Sem história estendida cadastrada.'}\n"
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

    cidades = data.get('cities', [])
    if cidades:
        c = cidades[0]
        content = f"CIDADE {c.get('name')} ({c.get('state')}): População {c.get('population')}, PIB {c.get('gdp')}, IDH {c.get('idh', 'N/A')}. Infraestrutura inclui {c.get('infrastructure', 'N/A')}"
        docs.append(Document(page_content=sanitize_text(content)))
    else:
        docs.append(Document(page_content="Nenhuma cidade no contexto."))

    for ed in data.get('condominiums', []):
        notas = ' '.join(ed.get('historicalNotes', []))
        content = f"EDIFÍCIO {ed.get('name')} em {ed.get('city')}. Notas: {notas}"
        docs.append(Document(page_content=sanitize_text(content)))

    return docs

# =====================================================================
# ⚡ INICIALIZAÇÃO DA IA NO STREAMLIT E FLUXO PRINCIPAL
# =====================================================================
@st.cache_resource(ttl=3600, show_spinner=False)
def setup_city_brain(nome_cidade, cidade_info):
    """Cria um cérebro Isolado e dedicado EXCLUSIVAMENTE para a cidade solicitada."""
    
    raw_data = fetch_all_entities()
    raw_data['cities'] = [cidade_info]
    
    cidade_slug = cidade_info.get('slug', '').lower().strip()
    cidade_alvo_norm = remove_accents(nome_cidade.lower().strip())
    
    def pertence_a_cidade(item):
        c = str(item.get('city', '')).lower().strip()
        if not c: return False
        return c == cidade_slug or remove_accents(c) == cidade_alvo_norm
        
    raw_data['condominiums'] = [ed for ed in raw_data.get('condominiums', []) if pertence_a_cidade(ed)]
    raw_data['constructors'] = [co for co in raw_data.get('constructors', []) if pertence_a_cidade(co)]
    raw_data['architects'] = [ar for ar in raw_data.get('architects', []) if pertence_a_cidade(ar)]
    
    local_cache = VerticalizaCache(raw_data)
    
    docs = build_documents(raw_data)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        groq_api_key=GROQ_API_KEY,
        max_retries=0,
        timeout=25
    )
    
    return local_cache, vectorstore, llm

todas_cidades = fetch_base_cities()

if not todas_cidades:
    st.error("🛑 Falha crítica ao carregar a lista de cidades base do Verticaliza. Verifique sua chave do Hygraph.")
    st.stop()

lista_nomes_cidades = [c['name'] for c in todas_cidades]

with st.sidebar:
    st.markdown("### ⚙️ Centro de Controle")
    st.markdown("A inteligência foca em uma cidade por vez para máxima precisão.")
    
    if "contexto_cidade" not in st.session_state:
        st.session_state.contexto_cidade = "Assis" if "Assis" in lista_nomes_cidades else lista_nomes_cidades[0]
        
    cidade_selecionada = st.selectbox("🏙️ Cidade Ativa no Radar", lista_nomes_cidades, index=lista_nomes_cidades.index(st.session_state.contexto_cidade))
    
    if cidade_selecionada != st.session_state.contexto_cidade:
        st.session_state.contexto_cidade = cidade_selecionada
        st.session_state.messages.append({"role": "assistant", "content": f"🔄 Mudei o meu radar de análise para a cidade de **{cidade_selecionada}**."})
        st.rerun()
        
    st.markdown("---")
    st.markdown("Acabou de cadastrar dados no Hygraph?")
    if st.button("🔄 Sincronizar Hygraph AGORA"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

st.title("🏛️ Verticaliza LUME")

boas_vindas_texto = f"""Olá! Eu sou o **LUME**, a Inteligência Artificial do projeto **Verticaliza**. 🏢✨

Atualmente, meus sensores estão focados na cidade de **{st.session_state.contexto_cidade}**.
Sou especialista em urbanismo, infraestrutura e no mercado imobiliário da região. 

Você pode me perguntar sobre:
- 🏙️ **Edifícios:** Fichas técnicas, história, andares e construtoras.
- 🏗️ **Construtoras:** Histórico e portfólio.
- 🌆 **Cidades:** Economia, cultura, hospitais, universidades e provedores.

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
    
    q_norm = remove_accents(pergunta.lower())
    mudou_cidade = False
    for cidade in lista_nomes_cidades:
         if remove_accents(cidade.lower()) in q_norm and cidade != st.session_state.contexto_cidade:
             st.session_state.contexto_cidade = cidade
             st.session_state.messages.append({"role": "assistant", "content": f"🔄 Notei que você perguntou sobre **{cidade}**. Mudando meu radar para lá..."})
             mudou_cidade = True
             st.rerun()

    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    cidade_info = next(c for c in todas_cidades if c['name'] == st.session_state.contexto_cidade)
    local_cache, vectorstore, llm = setup_city_brain(st.session_state.contexto_cidade, cidade_info)

    with st.chat_message("assistant"):
        with st.spinner("Aguarde..."):
            time.sleep(2.5)
            
            try:
                query_expandida = pergunta
                palavras_contexto = ['ele', 'ela', 'dele', 'dela', 'desse', 'dessa', 'quando', 'onde', 'arquiteto', 'engenheiro', 'projetou', 'plantas', 'universidade', 'faculdade', 'cultura', 'e', 'mas', 'não', 'nao', 'qual', 'quais', 'ultimo', 'ultima']
                if st.session_state.ultima_pergunta_usuario and (len(pergunta.split()) < 7 or any(p in remove_accents(pergunta.lower()).split() for p in palavras_contexto)):
                    query_expandida = f"{st.session_state.ultima_pergunta_usuario} | {pergunta}"

                dados_locais = local_cache.analyze_query(query_expandida)
                
                contexto_semantico = ""
                try:
                    documentos_relevantes = vectorstore.similarity_search(query_expandida, k=2)
                    contexto_semantico = "\n".join([doc.page_content for doc in documentos_relevantes])
                except Exception:
                    pass
                
                # 🚨 PROMPT BLINDADO - GROUNDING PROTOCOL 🚨
                prompt_sistema = (
                    f"Você é o LUME, a Inteligência Artificial especialista em urbanismo e mercado imobiliário. Seu radar atual está em {st.session_state.contexto_cidade}.\n"
                    "Aja como um consultor humano técnico, seco e estritamente baseado nos fatos.\n\n"
                    "PROTOCOLO DE SEGURANÇA CMS (SIGA RIGOROSAMENTE):\n"
                    "1. NUNCA MENCIONE SUAS FONTES: É TERMINANTEMENTE PROIBIDO usar palavras como 'CMS', 'Hygraph', 'Banco de dados', 'De acordo com...', 'Nos meus registros' ou 'Na tag'. Aja como se a informação estivesse na sua mente.\n"
                    "2. ALUCINAÇÃO ZERO (LIMITAÇÃO DE FATOS): É estritamente proibido deduzir, supor ou inventar: áreas de lazer (piscina, salão de festas), bairros, nomes de ruas, estilos arquitetônicos (ex: modernista), cursos de faculdades ou eventos turísticos que não estejam explicitamente marcados como '[FONTE CMS]'. Se o dado for '[DADO NÃO CATALOGADO]', você DEVE responder: 'Ainda não possuo essa informação catalogada.'\n"
                    "3. PROIBIÇÃO DE SÍNTESE ANALÍTICA: Se pedirem para cruzar com planos econômicos, governos ou agrupar por décadas/séculos, você DEVE recusar dizendo: 'O projeto Verticaliza foca em dados técnicos diretos e não realiza cruzamentos com planos econômicos governamentais ou análises por década.'\n"
                    "4. PROIBIDO PEDIR DESCULPAS: NUNCA peça desculpas por erros ou correções. Diga apenas 'Entendido. Atualizando a informação:' e continue.\n"
                    "5. PROIBIÇÃO DE ADJETIVOS CRIATIVOS: Não use adjetivos como 'belo', 'imponente', 'único', 'clássico', 'luxuoso' ou 'vibrante' se a descrição técnica não contiver essas palavras exatas.\n"
                    "6. IMAGENS E IDIOMA: Responda em Português do Brasil (proibido usar 'tu' ou 'fizestes'). Mantenha os blocos Markdown de imagens intocados.\n\n"
                    "--- SEU CONHECIMENTO (EXCLUSIVO) ---\n"
                    f"{dados_locais if dados_locais else ''}\n"
                    f"CONTEXTO ADICIONAL (Sempre confirme se a cidade citada no fragmento é {st.session_state.contexto_cidade}):\n"
                    f"{contexto_semantico if contexto_semantico else 'Nenhum'}\n"
                    "--- FIM DO SEU CONHECIMENTO ---"
                )
                
                mensagens = [SystemMessage(content=prompt_sistema)]
                
                for msg in st.session_state.messages[-5:]:
                    if msg["role"] == "user":
                        mensagens.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant" and not "🏢✨" in msg["content"] and not "🔄" in msg["content"]:
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