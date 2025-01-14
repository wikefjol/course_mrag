from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

def build_rag_chain(retriever, llm):

    template = """
    Use the following context to answer the question. Only use  information from the context provided.
    Do not ask questions.

    Context: {context}

    Question: {question}

    Answer: """

    prompt = PromptTemplate.from_template(template)

    doc_chain = retriever | (lambda docs: docs if docs else None)
    parser = StrOutputParser()
    rag_chain = (
        {"context": retriever | (lambda docs: " ".join(doc.page_content for doc in docs) if docs else ""),
        "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()

    )

    combined_chain = RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "answer": rag_chain,
            "docs": doc_chain,
        }
    )

    return combined_chain


def handle_query(rag_chain, query):
    try:
        return rag_chain.invoke(query)
    except Exception as e:
        # Log the error and optionally return a fallback response
        print(f"Error during chain invocation: {e}")
        return {"answer": "Sorry, something went wrong.", "docs": []}


def create_answer_fn(chain): #TODO: I want to move this function to the giskard_wrappers.py module, but in order to do that, we need to change the function so that it takes in a handle_query function and a rag_chain and then returns the answer_fn
    """
    Creates a wrapper function for querying the RAG chain.
    
    Args:
        chain: The RAG chain object.
    
    Returns:
        A callable function `answer_fn` compatible with Giskard library's interface.
    """
    def answer_fn(question, history=None):
        if not question:
            raise ValueError("Question cannot be None or empty.")
        
        chain_output = handle_query(chain, question)
        answer = chain_output.get("answer", "No answer available.")
        return answer
    
    return answer_fn
