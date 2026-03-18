import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HYGRAPH_URL = os.getenv("HYGRAPH_URL")
HYGRAPH_TOKEN = os.getenv("HYGRAPH_TOKEN")

def _hygraph_post(query: str, variables: dict | None = None):
    if not HYGRAPH_URL:
        return None, "Erro: HYGRAPH_URL não configurado."
    if not HYGRAPH_TOKEN:
        return None, "Erro: HYGRAPH_TOKEN não configurado."

    headers = {"Authorization": f"Bearer {HYGRAPH_TOKEN}"}
    try:
        response = requests.post(
            HYGRAPH_URL,
            json={"query": query, "variables": variables or {}},
            headers=headers,
            timeout=30,
        )
        res_json = response.json()
        if "errors" in res_json:
            return None, f"Erro Hygraph: {res_json['errors']}"
        return res_json.get("data"), None
    except Exception as e:
        return None, f"Erro de conexão: {e}"


def _hygraph_type_fields(type_name: str) -> list[str]:
    """
    Retorna os campos disponíveis de um type no schema do Hygraph via introspection.
    Se a introspection estiver bloqueada/der erro, retorna lista vazia.
    """
    introspection = """
    query($typeName: String!) {
      __type(name: $typeName) {
        fields { name }
      }
    }
    """
    data, err = _hygraph_post(introspection, {"typeName": type_name})
    if err or not data:
        return []
    fields = data.get("__type", {}).get("fields") or []
    return [f.get("name") for f in fields if f and f.get("name")]


def _pick_fields(available: list[str], desired: list[str], required: list[str] | None = None) -> list[str]:
    required = required or []
    avail = set(available)
    picked = [f for f in required if f in avail]
    picked += [f for f in desired if f in avail and f not in picked]
    return picked


def _build_verticaliza_query() -> str:
    city_fields = _pick_fields(
        _hygraph_type_fields("City"),
        desired=["state", "culture", "population", "gdp", "region", "slug"],
        required=["name"],
    )
    constructor_fields = _pick_fields(
        _hygraph_type_fields("Constructor"),
        desired=["city", "companyStatus", "status", "description", "slug"],
        required=["name"],
    )
    architect_fields = _pick_fields(
        _hygraph_type_fields("Architect"),
        desired=["city", "archStatus", "status", "specialties", "description", "slug"],
        required=["name"],
    )
    condominium_fields = _pick_fields(
        _hygraph_type_fields("Condominium"),
        desired=["city", "slug", "style", "floors", "yearBuilt", "completionYear", "height"],
        required=["name"],
    )
    event_fields = _pick_fields(
        _hygraph_type_fields("CulturalEvent"),
        desired=["date", "category", "description", "city", "slug"],
        required=["title"],
    )

    if not city_fields:
        city_fields = ["name"]
    if not constructor_fields:
        constructor_fields = ["name"]
    if not architect_fields:
        architect_fields = ["name"]
    if not condominium_fields:
        condominium_fields = ["name", "city"]
    if not event_fields:
        event_fields = ["title"]

    return f"""
    query {{
      cities {{ {' '.join(city_fields)} }}
      constructors {{ {' '.join(constructor_fields)} }}
      architects {{ {' '.join(architect_fields)} }}
      condominiums {{ {' '.join(condominium_fields)} }}
      culturalEvents {{ {' '.join(event_fields)} }}
    }}
    """


def get_verticaliza_context():
    query = _build_verticaliza_query()
    data, err = _hygraph_post(query)
    if err:
        print(f"❌ {err}")
        return err
    if not data:
        return "Erro: Resposta do Hygraph veio vazia."

    stats_cidades = {}
    for ed in data.get('condominiums', []):
        cid = ed.get('city', 'Indefinida')
        stats_cidades[cid] = stats_cidades.get(cid, 0) + 1

    doc = "# ACERVO VERTICALIZA\n\n"
    
    doc += "## RESUMO DE COBERTURA\n"
    for cidade, qtd in stats_cidades.items():
        doc += f"- {cidade}: {qtd} edifícios catalogados.\n"

    doc += "\n## CIDADES E CULTURA\n"
    for c in data.get('cities', []):
        name = c.get("name", "Indefinida")
        state = c.get("state")
        culture = c.get("culture", "Cultura não informada")
        if state:
            doc += f"- {name}/{state}: {culture}.\n"
        else:
            doc += f"- {name}: {culture}.\n"

    doc += "\n## EDIFÍCIOS E ARQUITETURA\n"
    for ed in data.get('condominiums', []):
        doc += f"- {ed.get('name', 'Edifício')} em {ed.get('city', 'Indefinida')}.\n"

    print(f"📊 Dados carregados com sucesso: {len(data.get('cities', []))} cidades.")
    return doc

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🏛️  VERTICALIZA LUME - ENGINE ATIVADA")
    print("="*50)
    
    # Carrega o contexto UMA VEZ
    contexto_global = get_verticaliza_context()
    
    if contexto_global.startswith("Erro"):
        print(f"🛑 Falha crítica ao carregar dados. Detalhes: {contexto_global}")
    else:
        while True:
            pergunta = input("\n👤 VOCÊ: ")
            
            if pergunta.lower() in ['sair', 'exit', 'quit']:
                break
                
            print("🤖 LUME analisando horizontes...")
            
            llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Você é o LUME. Use APENAS os dados fornecidos. "
                           "Se não houver dados de uma cidade no contexto, diga que ainda não a catalogamos. "
                           "Contexto:\n{contexto}"),
                ("user", "{pergunta}")
            ])
            
            chain = prompt | llm
            res = chain.invoke({"contexto": contexto_global, "pergunta": pergunta})
            print(f"\n✨ LUME: {res.content}")