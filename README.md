LUME 🏛️

Language, Urbanism, Memory, and Engineering

LUME is a specialized Urban Intelligence Engine developed for the Verticaliza Project. It serves as a Senior Technical Consultant for real estate analytics, urbanism, and local history, bridging the gap between a Headless CMS (Hygraph) and High-Performance LLMs via Groq Cloud.

🚀 The Core Philosophy: Deterministic Grounding

Unlike generic AI chatbots, LUME is built on a "Python-First, AI-Second" architecture. It prioritizes data integrity over generative creativity, ensuring that every piece of information is anchored in verified records.

Key Technical Features

City-Agnostic Engine: The system dynamically reconfigures its "contextual brain" based on the selected city (Assis, Londrina, etc.), filtering entities via normalized slugs to prevent data cross-contamination.

Mathematical Guardrails: Sorting, ranking (tallest buildings, oldest structures), and market statistics are calculated via deterministic Python logic before reaching the LLM. This prevents "hallucination-by-synthesis" where models often fail at basic numerical comparisons.

Hybrid RAG (Retrieval-Augmented Generation):

Structured Data: Direct mapping of GraphQL objects (PIB, IDH, Infrastructure, University Courses).

Unstructured Data: FAISS Vector Store integration for processing richContent (historical newspaper archives and long-form notes).

Multilingual Intent Recognition: Native support for Portuguese, English, Spanish, and German intent triggers, allowing global real estate investors to query local data naturally.

🛡️ The Grounding Protocol (Strict Guardrails)

To ensure LUME remains a reliable technical tool for professional use, we implemented a strict security layer:

Zero-Amenity Hallucination: If "Pool", "Gym", or "Balcony" is not explicitly in the CMS record, the engine injects a hard block forbidding the AI from "filling the gaps" with generic descriptions.

Strict Source Attribution: Only data tagged and verified by the engine is used for factual claims. If the data is missing, the model is instructed to state the absence of records rather than guessing.

Consultant Persona: The model is prohibited from using "AI-speak" (e.g., "according to my database"). It maintains a professional "Senior Consultant" authority, speaking as if it possesses the local knowledge natively.

🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| Language | Python 3.11 |
| AI Orchestration | LangChain |
| LLM Provider | Groq Cloud (Llama-3-8B / 70B) |
| Interface | Streamlit |
| Vector Database | FAISS |
| Embeddings | HuggingFace Multilingual (sentence-transformers) |
| Data Source | Hygraph (GraphQL Headless CMS) |

📦 Deployment & Execution

Local Setup

Clone the repository:

```bash
git clone https://github.com/your-repo/verticaliza-lume.git
cd verticaliza-lume
```

Configure Environment:
Create a `.env` file in the root directory:

```bash
GROQ_API_KEY=your_key_here
HYGRAPH_URL=your_endpoint_here
HYGRAPH_TOKEN=your_token_here
```

Install Dependencies:

```bash
pip install -r requirements.txt
```

Run the Application:

```bash
streamlit run verticaliza_lume.py
```

Docker Implementation

Build the production-ready image:

```bash
docker build -t verticaliza-lume .
```

Run the container:

```bash
docker run -p 8501:8501 --env-file .env verticaliza-lume
```

👨‍💻 Developer Note

This project demonstrates the implementation of a Reliable AI Assistant in a niche market. The core challenge solved was not making the AI talk, but making it remain silent when data is unavailable, ensuring 100% fidelity to the Verticaliza database.
