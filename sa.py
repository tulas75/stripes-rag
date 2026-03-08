# Load environment variables (including HF_TOKEN)
from dotenv import load_dotenv
load_dotenv()

import os
import sys

#from phoenix.otel import register
#from openinference.instrumentation.smolagents import SmolagentsInstrumentor

#os.environ["PHOENIX_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.eN7OrwgzyHBp87NU4DCYRwmuWZVSYblBBpM5F077EBg"
#os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com/s/gnucoop"

#register(project_name="smol-rag")
#SmolagentsInstrumentor().instrument()

# Connect to the vector store using stripes-rag infrastructure
print("🔧 Setting up embeddings...")
from stripes_rag.db import get_engine, get_vectorstore, init_vectorstore_table
from stripes_rag.embeddings import get_embeddings

engine = get_engine()
init_vectorstore_table(engine)
embeddings = get_embeddings()

print("🔧 Connecting to PGVectorStore...")
vectorstore = get_vectorstore(engine, embeddings)

print("✅ Connected to vector store successfully")

from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Searches the organizational knowledge base for relevant past projects, "
        "experiences, methodologies, and lessons learned. Call this tool multiple "
        "times with different queries to gather broad context from different angles "
        "(e.g., technical approaches, stakeholder concerns, similar past projects, "
        "risks encountered)."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query. Use varied queries to explore different facets of the topic.",
        }
    }
    output_type = "string"

    def __init__(self, vectorstore, **kwargs):
        super().__init__(**kwargs)
        self.vectorstore = vectorstore

    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "Your search query must be a string"

        # Retrieve relevant documents with scores using similarity_search_with_score
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=10)

        # Format the retrieved documents for readability
        result = "\nRetrieved documents:\n"
        for i, (doc, score) in enumerate(docs_with_scores):
            # Convert distance to similarity (since using cosine distance)
            similarity = 1 - score
            source = doc.metadata.get("source_file", "unknown")
            headings = doc.metadata.get("headings", "")
            pages = doc.metadata.get("page_numbers", "")
            result += (
                f"\n\n===== Document {i} (similarity: {similarity:.4f}) =====\n"
                f"Source: {source}\n"
            )
            if headings:
                result += f"Headings: {headings}\n"
            if pages:
                result += f"Pages: {pages}\n"
            result += f"\n{doc.page_content}"
        
        return result

# Initialize our retriever tool with the vector store
retriever_tool = RetrieverTool(vectorstore)

from smolagents import CodeAgent, LiteLLMModel

#os.environ["OLLAMA_API_BASE"] = ""

#model_id = "deepseek/deepseek-chat"
#model_id = "cerebras/llama3-70b-instruct"
#model_id = "deepinfra/mistralai/Mistral-Small-3.2-24B-Instruct-2506"
#model_id = "deepinfra/mistralai/Mistral-Small-24B-Instruct-2501"
#model_id = "hosted_vllm/allenai/Olmo-3-7B-Instruct"
#model_id = "deepinfra/Qwen/Qwen3-Next-80B-A3B-Instruct"
#model_id = "deepinfra/Qwen/Qwen3-30B-A3B"
#model_id = "anthropic/MiniMax-M2.1"
#model_id = "ollama_chat/qwen3.5:35b"
#model_id = "ollama_chat/qwen3:4b-instruct-2507-q8_0"
#model_id = "ollama_chat/qwen3.5:9b-q8_0"
model_id = "ollama_chat/qwen3.5:4b-q8_0"
#model_id = "groq/llama-3.1-8b-instant"
#model_id = "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"
#model_id = "mistral/mistral-small-latest"
#model_id = "fireworks_ai/accounts/fireworks/models/qwen3-vl-30b-a3b-instruct"
authorized_imports = ["json"]

# Initialize the agent with our retriever tool
agent = CodeAgent(
    tools=[retriever_tool],
    model=LiteLLMModel(model_id=model_id, temperature=0.3),  # slight creativity for narrative
    max_steps=6,  # allow multiple retrieval passes
    stream_outputs=True,
    additional_authorized_imports=authorized_imports,
    verbosity_level=2,
)

language="ITA"

additional_notes = """
You are a "Project Architect" — a senior advisor who helps users design
and plan new projects by drawing on an organizational knowledge base of
past experiences, methodologies, and lessons learned.

APPROACH
- Always call the `retriever` tool BEFORE answering.
- Use multiple retrieval queries from different angles:
  1) Direct topic search
  2) Related methodologies or frameworks
  3) Lessons learned, risks, or pitfalls
- Base your response ONLY on retrieved information. Do NOT rely on
  general pre-trained knowledge.

RESPONSE STYLE
- Write in a narrative, consultative tone — like a senior advisor
  briefing a project team.
- Be expansive and rich in context. Explain trade-offs, connect ideas
  across sources, and suggest alternatives.
- Reference the source documents when making claims.
- Structure your response with these sections:
  ## Background & Context
  ## Relevant Past Experience
  ## Recommended Approach
  ## Risks & Lessons Learned
  ## Suggested Next Steps

OUTPUT FORMAT
- Return your response as a single markdown-formatted string (NOT JSON).
- At the end, add a "## Follow-up Questions" section with 2-3
  questions the user might explore next.

LANGUAGE
- Always respond in {language}.
"""


# Parse command line arguments
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py \"Your question here\"")
        print("\nExample:")
        print('  python script.py "What is the difference between forward and backward pass?"')
        sys.exit(1)
    
    # Get the question from command line arguments
    question = " ".join(sys.argv[1:])
    
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print('='*80)
    
    try:
        agent_output = agent.run(question,additional_args=dict(additional_notes=additional_notes))
        print("\n📝 FINAL ANSWER:")
        print(agent_output)
        #print(agent.monitor)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
