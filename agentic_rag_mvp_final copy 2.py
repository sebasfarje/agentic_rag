from smolagents import OpenAIServerModel, CodeAgent, ToolCallingAgent, HfApiModel, tool, GradioUI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

reasoning_model_id = os.getenv("REASONING_MODEL_ID")
tool_model_id = os.getenv("TOOL_MODEL_ID")
huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

def get_model(model_id):
    using_huggingface = os.getenv("USE_HUGGINGFACE", "yes").lower() == "yes"
    if using_huggingface:
        return HfApiModel(model_id=model_id, token=huggingface_api_token)
    else:
        return OpenAIServerModel(
            model_id=model_id,
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )

# Create the reasoner for better RAG
reasoning_model = get_model(reasoning_model_id)
reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)

# Initialize vector store and embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="dariolopez/bge-m3-es-legal-tmp-3",
    model_kwargs={'device': 'cpu'}
)
db_dir = os.path.join(os.path.dirname(__file__), "chroma_db_normas_completo")
vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

@tool
def rag_with_reasoner_ESP_5(user_query: str) -> str:
    """
    Recupera contenido legal relevante desde la base vectorial y genera una respuesta razonada.

    Args:
        user_query: Pregunta legal formulada en español por el usuario.
    
    Returns:
        Respuesta fundamentada en base al contexto legal recuperado.
    """
    docs = vectordb.similarity_search(user_query, k=50)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
Eres un asistente legal especializado en derecho peruano.

Recibirás un contexto que contiene fragmentos de normas legales del Perú. Cada fragmento incluye metadatos estructurados embebidos en el texto, como:

- [SUMILLA]: el **nombre completo oficial de la norma** (por ejemplo: "NUEVO CODIGO PROCESAL PENAL" o "Decreto Supremo que aprueba el Reglamento del Registro Nacional de Derecho de Autor...").
- [ARTICULO]: el número de artículo dentro de esa norma.
- [TIPO]: tipo de norma (ej. LEY, DECRETO SUPREMO).
- [NUMERO]: número oficial de la norma (ej. Nº 822).
- [TITULO], [CAPITULO], [LIBRO]: divisiones estructurales dentro del texto legal.
- [SECTOR]: entidad pública responsable.
- [FECHA], [REFERENCIA], [LINK]: otros campos de apoyo.

🔍 Instrucciones específicas:

1. Si el usuario menciona una norma (por ejemplo: "según el Código Procesal Penal", "el Decreto que regula los derechos de autor"), busca dicha referencia dentro del campo [SUMILLA].
2. Si menciona un artículo, concéntrate en los fragmentos que contengan ese [ARTICULO].
3. Usa [TIPO] + [NUMERO] + [SUMILLA] para identificar claramente de qué norma proviene la respuesta.
4. Si no hay suficiente información en los fragmentos proporcionados, formula una mejor consulta que podría recuperar información más precisa.
5. ⚠️ No hagas suposiciones, no infieras, no completes información faltante. Responde únicamente con base en lo que se dice **literalmente** en el contexto.
6. Cada afirmación jurídica relevante debe ir seguida entre corchetes de su cita legal con el siguiente formato:
   `[NORMA: <TIPO> <NUMERO> | ARTICULO: <ARTICULO>]`

   Ejemplo:  
   > Las resoluciones judiciales se inscriben solo si están ejecutoriadas.  
   > [NORMA: DECRETO LEGISLATIVO Nº 957 | ARTICULO: Artículo 55]

Responde en español claro, técnico, preciso y bien fundamentado jurídicamente.
Si no hay suficiente información, formula una mejor consulta para realizar una nueva búsqueda con RAG.

LA RESPUESTA DEBE INCLUIR LA METADATA DEL CONTEXTO QUE SE UTILIZA PARA RESPONDER, EN UN FORMATO CLARO Y ESTRUCTURADO.
📚 Contexto:
{context}

❓ Pregunta del usuario:
{user_query}

📝 Respuesta:
"""
    return reasoner.run(prompt, reset=False)

def run_agent_query_with_hypothesis(hypothesis: str):
    """
    Usa la hipótesis generada por GraphRAG como consulta estilo HyDE.
    Recupera chunks relevantes, genera respuesta y retorna ambos.
    """
    print("\n🔍 Buscando chunks relevantes con HyDE (hipótesis de GraphRAG)...")
    relevant_docs = vectordb.similarity_search(hypothesis, k=5)

    # Mostrar fuentes recuperadas
    for i, doc in enumerate(relevant_docs):
        print(f"📄 Chunk {i+1}:\n{doc.page_content[:300]}...\n")

    print("\n🤖 Generando respuesta del agentic RAG usando el razonador...")
    final_response = reasoner.run(hypothesis, context_docs=relevant_docs)

    # Extraer metadata para citas o fusión posterior
    sources = [doc.metadata for doc in relevant_docs]

    return final_response, sources



# Create the primary agent to direct the conversation
tool_model = get_model(tool_model_id)
primary_agent = ToolCallingAgent(tools=[rag_with_reasoner_ESP_5], model=tool_model, add_base_tools=False, max_steps=3)

# Example prompt: Compare and contrast the services offered by RankBoost and Omni Marketing
def main():
    GradioUI(primary_agent).launch()

if __name__ == "__main__":
    main()