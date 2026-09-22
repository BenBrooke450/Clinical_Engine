



from _1_Med_Pinecone_Embedding import process_patient_data, patient_passed_processed
from _2_Embed_rate_pinecone import process_patient_review
from _3_Vector_database import process_pinecone_embeddings
from _4_RAG_pipeline import query_patient_rag

#ROLES ["RESEARCHER","SENIOR_RESEARCHER"]

patient = 10  #10,11 known subject  #

if patient_passed_processed(patient):
    print("== Subject has already been processed ==")
    pass
else:
    print("== Subject has not been processed ==")

    process_patient_data(patient)

    process_patient_review(patient)

    process_pinecone_embeddings(patient)



query = "Find patient 761839 who went to Hangzhou First People's Hospital, bring me back all the information"

answer, docs = query_patient_rag(query=query, user_role="RESEARCHER")

print(answer)
print(docs)





