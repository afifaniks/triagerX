import pandas as pd
import requests

df_test = pd.read_csv("/home/mdafifal.mamun/notebooks/triagerX/openj9_test_17.csv")
df_test = df_test[df_test["component"].notna()]

url = "http://127.0.0.1:8000/recommendation"

top_1_correct_owners = 0
top_3_correct_owners = 0

top_1_correct_components = 0
top_3_correct_components = 0

total_issues = len(df_test)

for index, row in df_test.iterrows():
    print(f"Test index: {index}")
    payload = {
        "issue_title": str(row["issue_title"]),
        "issue_description": str(row["description"]),
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        recommended_developers = response.json()["recommended_developers"]
        recommended_components = response.json()["recommended_components"]

        if row["owner"] == recommended_developers[0]:
            top_1_correct_owners += 1
        if row["owner"] in recommended_developers[:3]:
            top_3_correct_owners += 1

        if row["component"] == recommended_components[0]:
            top_1_correct_components += 1
        if row["component"] in recommended_components[:3]:
            top_3_correct_components += 1

        print(
            f"Recommedned developers: {recommended_developers[:3]}, Actual: {row['owner']}"
        )
        print(
            f"Recommedned components: {recommended_components[:3]}, Actual: {row['component']}"
        )
    else:
        print(
            f"Failed to get recommendation for issue {row['issue_number']} (status code: {response.status_code})"
        )

top_1_owner_accuracy = top_1_correct_owners / total_issues * 100
top_3_owner_accuracy = top_3_correct_owners / total_issues * 100

top_1_component_accuracy = top_1_correct_components / total_issues * 100
top_3_component_accuracy = top_3_correct_components / total_issues * 100

print(f"Top-1 Owner Accuracy: {top_1_owner_accuracy:.2f}%")
print(f"Top-3 Owner Accuracy: {top_3_owner_accuracy:.2f}%")

print(f"Top-1 Component Accuracy: {top_1_component_accuracy:.2f}%")
print(f"Top-3 Component Accuracy: {top_3_component_accuracy:.2f}%")
