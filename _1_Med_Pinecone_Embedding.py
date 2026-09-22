def process_patient_data(subject_number: int = 10):
    from datasets import Dataset
    import pandas as pd
    import json
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage

    data = "/Users/benjaminbrooke/.cache/huggingface/datasets/homeway___medical_rag/default/0.0.0/1c81373d9f6ab4c494ea6ee5697c408abc78a118/medical_rag-test.arrow"

    dataset = Dataset.from_file(data)
    df = dataset.to_pandas()
    pd.set_option("display.max_colwidth", None)

    med_prac_df = df[df["role"] == "Medical Practitioners"]

    first_patient = med_prac_df.iloc[0]


    def check_data_patient(number: int):
        f_p = med_prac_df.iloc[number]
        return f_p["document"]

    llm = ChatOllama(model="qwen3:1.7b", temperature=0)
    llm_judge = ChatOllama(model="qwen3:1.7b", temperature=0)

    prompt_template_med_JSON = ChatPromptTemplate.from_template("""
            You are an information extraction system.

            Extract the patient information from the provided clinical text and
            separate it into two categories: personal data and medical data.

            The personal data section normally/usually appears before "Prosecution".
            The medical examination begins with "Prosecution" and continues through
            the remainder of the clinical record. 

            USE "unclassified_text" to put any data you can't classify and DO not leave any data out.

            Return using this JSON structure:

            {{
                "patient": {{
                    "name": null,
                    "age": null,
                    "sex": null,
                    "id_number": null,
                    "contact_number": null,
                    "address": null
                }},
                "medical": {{
                    "hospital": null,
                    "department": null,
                    "date": null,
                    "symptoms": null,
                    "medical_history": null,
                    "diagnosis": null,
                    "treatment_plan": null
                }},
                "unclassified_text": {{
                }}
            }}

            Rules:
            - Do not invent information.
            - If a field is not present, return null.
            - Preserve the original meaning of the clinical information.
            - Do not modify or infer patient identifiers.
            - Return valid JSON only.

            Clinical text:
            {context}
            """)

    prompt_template_LLM_as_judge = ChatPromptTemplate.from_template("""

            You are an LLM-as-a-Judge responsible for validating the completeness and accuracy of an LLM's structured extraction.

            Your task is to compare the LLM output against the original unstructured clinical text and identify whether any information has been missed.

            Rules:

            Cross-check the entire LLM output against the entire clinical text.
            Identify any information present in the clinical text that is missing from the LLM output.
            Do not discard any information.

            Do not invent or infer information that is not explicitly present in the clinical text.

            If information cannot be confidently classified into an existing field, place it under unclassified_text.

            If multiple pieces of information are missing, ensure that ALL of them are captured.

            Preserve the original meaning and values of the clinical text.

            either in an appropriate field or under unclassified_text.

            ADD what is missing to the "unclassified_text": {{}} field or an appropriate field, RETURN IT in a JSON format

            LLM output:
            {LLM_output}

            Clinical text:
            {context}
            """)

    print("\n ============= STAGE 0 ============= \n ")

    context = check_data_patient(subject_number)
    messages = prompt_template_med_JSON.format_messages(context=context)
    answer = llm.invoke(messages)

    print(f"\n Answer:{answer.content} \n")

    print("\n ============= STAGE 1 ============= \n ")

    messages = prompt_template_LLM_as_judge.format_messages(context=context, LLM_output=answer.content)
    judge_answer = llm_judge.invoke(messages)

    print(f"\n Answer:{judge_answer.content} \n")

    with open(f"/Users/benjaminbrooke/Downloads/JSON_MED/Subjects/judge_answer_{subject_number}.json", "w") as f:
        json.dump({"content": judge_answer.content, "metadata": judge_answer.response_metadata}, f, indent=4,
                  default=str)

    return judge_answer.content


def patient_passed_processed(patient_id: int, folder_path= "/Users/benjaminbrooke/Downloads/JSON_MED/Subjects") -> bool:
    import os

    target_prefix = f"judge_answer_{patient_id}"

    if not os.path.exists(folder_path):
        return False

    for file in os.listdir(folder_path):
        if file.startswith(target_prefix):
            return True

    return False


if __name__ == "__main__":
    process_patient_data(10)