"""
RAG chain: retriever → prompt → Ollama LLM → answer.
"""

from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, TOP_K

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the following conversation and a follow up question,
rephrase the follow up question to be a standalone question in the same language.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""
)

QA_PROMPT = PromptTemplate.from_template(
    """You are a helpful customer-support assistant for STAR Assurances.
Answer the user's question using ONLY the context below.
If the context does not contain the answer, say you don't have that information
and suggest the user contact STAR Assurances directly.
Always reply in the same language the user used (French or Arabic or English).

Context:
{context}

Question: {question}

Helpful answer:"""
)


def build_rag_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    """Build and return a conversational RAG chain."""

    # --- LLM ------------------------------------------------------------------
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0.3,
    )

    # --- Retriever ------------------------------------------------------------
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # --- Memory ---------------------------------------------------------------
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5,  # keep last 5 exchanges
    )

    # --- Chain ----------------------------------------------------------------
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )
    return chain
