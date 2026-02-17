from multiprocessing import context
from langchain_core.prompts import ChatPromptTemplate
from documents import vector_store
from models import get_llm
from api_keys import set_keys
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_community.retrievers.arxiv import ArxivRetriever
from langchain_community.retrievers.tavily_search_api import TavilySearchAPIRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
import operator
from typing import Annotated, TypedDict
from langgraph.graph import START, StateGraph, END, add_messages
from threads import pick_or_create_thread, checkpointer

system_prompt = """You are a helpful assistant that answers questions based on the provided context and your internal knowledge.
For each question, you should use the retrieved context and your internal knowledge to provide a comprehensive answer. If the context does not contain relevant information, rely on your internal knowledge to answer the question.
For each piece of information you use from the context, cite the source in parentheses from the provided source Example: (Source: <document.pdf>) or (Source: <hyperlink>).
If the fact is from your internal knowledge, cite it as (Source: Internal Knowledge).
If you don't know the answer, say you don't know."""

document_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
wiki_retriever = WikipediaRetriever()
arxiv_retriever = ArxivRetriever()
web_search_retriever = TavilySearchAPIRetriever()

set_keys()

class SessionState(TypedDict):
    needs_context: bool
    context: Annotated[list[str], operator.add]
    answer: str
    messages: Annotated[list, add_messages]
    search_documents: bool
    search_wikipedia: bool
    search_arxiv: bool
    search_web: bool


def needs_context(state: SessionState):
    existing_context = "\n\n".join(state.get("context", [])) or "No context available"
    prompt = f"""You are an expert context verifier. Given the existing context and a new question, determine if the question can be answered with the existing context or if additional information is needed.
    Existing Context: {existing_context}
    New Question: {state["messages"][-1].content}
    Does the question require additional context to answer accurately?  
    **Respond with 'Yes' or 'No' only.**
    """
    response = get_llm().invoke(prompt)
    print(f"LLM response for needs_context: {response.content}")
    if response.content.strip().lower() == "yes":
        return {"needs_context" : True}
    else:
        return {"needs_context" : False}

def needs_context_condition(state: SessionState):
    if state["needs_context"]:
        return "get_context"
    else:
        return "generate_answer"

def get_context(state: SessionState):
    search_documents = state.get("search_documents", True)
    search_wikipedia = state.get("search_wikipedia", True)
    search_arxiv = state.get("search_arxiv", True)
    search_web = state.get("search_web", True)
    print("Retrieving context for question:", state["messages"][-1].content)
    query = state["messages"][-1].content

    def safe_invoke(retriever, label: str):
        try:
            print(f"Invoking {label} retriever...")
            return retriever.invoke(query)
        except Exception as exc:
            print(f"{label} retriever failed: {exc}")
            return []

    doc_results = safe_invoke(document_retriever, "Documents") if search_documents else []
    wiki_results = safe_invoke(wiki_retriever, "Wikipedia") if search_wikipedia else []
    arxiv_results = safe_invoke(arxiv_retriever, "Arxiv") if search_arxiv else []
    web_search_results = safe_invoke(web_search_retriever, "Web") if search_web else []

    for result in arxiv_results:
        result.metadata["source"] = result.metadata["Entry ID"]
    
    # Combine and format results as needed
    full_context = "\n".join(
        [
            doc.page_content + " Source: " + doc.metadata["source"] if "source" in doc.metadata else doc.page_content + " Source: Unknown"
            for doc in doc_results + wiki_results + arxiv_results + web_search_results
        ]
    )
    compression_prompt = f"""Given the following context, only keep the information that is relevant to answer the question.
        
    Context: {full_context}
    
    Question: {query}
    
    - Remove any irrelevant details.
    - Remove any duplicate information.
    - Remove any formatting or metadata that is not necessary for understanding the content.
    - DO NOT remove the original source information which is formatted as (Source: ) in the context.
    - Keep the source information with the content from that source. like this example: "The Eiffel Tower is located in Paris. (Source: https://en.wikipedia.org/wiki/Paris)"
    - Provide the compressed context without any additional commentary."""
    
    compressed_context = get_llm().invoke(compression_prompt).content.strip()

    return {"context": [compressed_context]}

def generate_answer(state: SessionState):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Context:{context} \n\nQuestion: {question}")
    ]
    )
    context_text = "\n\n".join(state.get("context", [])) or "No context available"
    input = prompt.format(context=context_text, question=state["messages"][-1].content )
    print(f"Prompt for answer generation:\n{input}\n")
    answer = get_llm().invoke(input)
    return {"answer": answer, "messages": [answer]}

rag_graph = StateGraph(SessionState)
rag_graph.add_node("needs_context", needs_context)
rag_graph.add_node("get_context", get_context)
rag_graph.add_node("generate_answer", generate_answer)
rag_graph.add_edge(START, "needs_context")
rag_graph.add_conditional_edges("needs_context", needs_context_condition, ["get_context", "generate_answer"])
rag_graph.add_edge("get_context", "generate_answer")
rag_graph.add_edge("generate_answer", END)
rag_graph_compiled = rag_graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = pick_or_create_thread()
    print("Type your questions below. Type 'quit' to exit, 'switch' to change threads.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "switch":
            config = pick_or_create_thread()
            continue

        result = rag_graph_compiled.invoke(
            {"messages": [("human", user_input)]},
            config=config,
        )
        print(f"\nAssistant: {result['answer'].content}\n")

