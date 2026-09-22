



def process_patient_review(subject_number: int = 10):
    import json
    import random
    import copy

    with open(f"/Users/benjaminbrooke/Downloads/JSON_MED/Subjects/judge_answer_{subject_number}.json", "r") as f:
        saved_answer = json.load(f)

    data = json.loads(saved_answer["content"])

    random_number = random.randint(10**5, (10**6) - 1)

    patient_confidential = copy.deepcopy(data)
    patient_confidential["patient"]["Assigned_Randomized_Subject_Number"] = random_number
    patient_confidential["patient"]["STATUS"] = "CONFIDENTIAL"

    patient_reviewable = copy.deepcopy(data)

    NULL_FIELDS = ["name", "address", "id_number", "contact_number"]

    for field in NULL_FIELDS:
        if field in patient_reviewable["patient"]:
            patient_reviewable["patient"][field] = None

    patient_reviewable["patient"]["Assigned_Randomized_Subject_Number"] = random_number
    patient_reviewable["patient"]["STATUS"] = "REVIEWABLE"

    print("\n ========== CONFIDENTIAL ========== \n")
    print("Randomized Subject Number: ", random_number)
    print(json.dumps(patient_confidential, indent=4))

    print("\n ========== REVIEWABLE ========== \n")
    print("Randomized Subject Number: ", random_number)
    print(json.dumps(patient_reviewable, indent=4))


    with open(f"/Users/benjaminbrooke/Downloads/JSON_MED/CONFIDENTIAL/judge_answer_{subject_number}_CONFIDENTIAL.json", "w") as f:
        json.dump({"content": patient_confidential}, f, indent=4, default=str)

    with open(f"/Users/benjaminbrooke/Downloads/JSON_MED/REVIEWABLE/judge_answer_{subject_number}_REVIEWABLE.json", "w") as f:
        json.dump({"content": patient_reviewable}, f, indent=4, default=str)

    return patient_confidential, patient_reviewable


if __name__ == "__main__":
    process_patient_review(10)