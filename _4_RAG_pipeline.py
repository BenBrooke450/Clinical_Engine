



def query_patient_rag(query: str, user_role: str = "RESEARCHER"):
    import os
    from dotenv import load_dotenv
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_pinecone import PineconeVectorStore
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    load_dotenv()

    ROLE_ACCESS = {"RESEARCHER": ["REVIEWABLE"],"SENIOR_RESEARCHER": ["REVIEWABLE", "CONFIDENTIAL"]}

    allowed_namespaces = ROLE_ACCESS.get(user_role, ["REVIEWABLE"])

    llm = ChatOllama(model="qwen3:1.7b")
    embedding = OllamaEmbeddings(model="embeddinggemma")

    vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"),embedding=embedding,pinecone_api_key=os.getenv("PINECONE_API_KEY"))

    if "CONFIDENTIAL" in allowed_namespaces:
        retrieve = vectorstore.as_retriever(search_kwargs={"k": 2, "namespace": "CONFIDENTIAL"})
    else:
        retrieve = vectorstore.as_retriever(search_kwargs={"k": 2, "namespace": "REVIEWABLE"})

    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:

        {context}

        Question: {question}

        Provide a detailed answer:
        """
    )

    docs = retrieve.invoke(query)
    context = "\n\n".join(document.page_content for document in docs)
    messages = prompt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)

    return response.content, docs


if __name__ == "__main__":
    query_str = "Find patient 944668 who went to Hangzhou First People's Hospital, bring me back all the information"

    answer, docs = query_patient_rag(query_str, user_role="RESEARCHER")

    print(answer)
    print(docs)