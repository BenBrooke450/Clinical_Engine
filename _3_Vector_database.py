



def process_pinecone_embeddings(patient_id,base_path: str = "/Users/benjaminbrooke/Downloads/JSON_MED"):
    from _1_Med_Pinecone_Embedding import patient_passed_processed
    import os
    from dotenv import load_dotenv
    from langchain_text_splitters import CharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from langchain_pinecone import PineconeVectorStore
    from langchain_ollama import OllamaEmbeddings

    load_dotenv()

    embeddings = OllamaEmbeddings(model="embeddinggemma")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    target_prefix = f"judge_answer_{patient_id}"

    for namespace in ["CONFIDENTIAL", "REVIEWABLE"]:
        folder = os.path.join(base_path, namespace)

        print(f"\n===== Processing {namespace} =====")

        for filename in os.listdir(folder):
            if not (filename.startswith(target_prefix) and filename.endswith(".json")):
                print(f"Subject {filename} has already been stored in a vector database")
                continue

            file_path = os.path.join(folder, filename)
            print(f"Loading: {filename}")

            loader = TextLoader(file_path=file_path)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)

            for document in texts:
                document.metadata["access_level"] = namespace
                document.metadata["source_file"] = filename

                if namespace == "CONFIDENTIAL":
                    document.metadata["required_role"] = "SENIOR_RESEARCHER"
                elif namespace == "REVIEWABLE":
                    document.metadata["required_role"] = "RESEARCHER"


            print("Processing: VectorDatabase Embedding")

            PineconeVectorStore.from_documents(texts, embedding=embeddings, index_name=os.environ["INDEX_NAME"], namespace=namespace)



if __name__ == "__main__":
    process_pinecone_embeddings()