from pathlib import Path
import json
import re

# Define the directory path
dir_path = Path("/home/mdafifal.mamun/notebooks/triagerX/test/eval_data")

# Initialize counters for accuracy calculations
top1_correct = 0
top2_correct = 0
top3_correct = 0
total_issues = 0

considered_labels = {
    'comp:build': 0, 'comp:gc': 1, 'comp:infra': 2, 'comp:jclextensions': 3, 
    'comp:jit': 4, 'comp:jitserver': 5, 'comp:jvmti': 6, 'comp:test': 7, 'comp:vm': 8
}

def extract_actual_and_predicted_components(issue):
    label_keys = list(considered_labels.keys())
    correct_labels = [label["name"] for label in issue["labels"]]
    correct_labels = [label for label in correct_labels if label in label_keys]
    timeline_data = issue["timeline_data"]
    bot_comment = [event for event in timeline_data if event.get("user", {}).get("login", None) == "github-actions[bot]"][0]["body"]

    predicted_components = re.findall(r'comp:\w+', bot_comment)
    return correct_labels, predicted_components

# Iterate through all JSON files in the directory
for file in dir_path.glob("*.json"):
    try:
        with file.open("r", encoding="utf-8") as f:
            issue = json.load(f)
            correct_labels, predicted_components = extract_actual_and_predicted_components(issue)
            if len(correct_labels) == 0:
                print("Doesn't have a supported component!")
                continue

            total_issues += 1

            if predicted_components[0] in correct_labels:
                top1_correct += 1
            if set(predicted_components[:2]).intersection(correct_labels):
                top2_correct += 1
            if set(correct_labels).intersection(predicted_components):
                top3_correct += 1
            else:
                print("No match: ", issue["number"])

    except json.JSONDecodeError as e:
        print(f"Failed to load {file.name}: {e}")

# Calculate and print accuracies
if total_issues > 0:
    print(f"Total files loaded: {total_issues}")
    print(f"Top-1 Accuracy: {top1_correct / total_issues:.2%}")
    print(f"Top-2 Accuracy: {top2_correct / total_issues:.2%}")
    print(f"Top-3 Accuracy: {top3_correct / total_issues:.2%}")
else:
    print("No valid issues were loaded.")