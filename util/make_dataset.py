import json
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm


class IssueExtractor:
    def extract_issues(self, json_root):
        all_issues = []
        file_paths = os.listdir(json_root)
        progress_bar = tqdm(
            total=len(file_paths), desc="Processing issues", unit="files", leave=False
        )
        for file_path in file_paths:
            progress_bar.update(1)
            progress_bar.set_description(desc=f"Processing file: {file_path}")
            with open(os.path.join(json_root, file_path), "r") as file:
                issue = json.load(file)
                if "/pull/" in issue["html_url"]:
                    print("Pull data")
                    continue
                all_issues.append(self.extract_issue_details(issue))

        return all_issues

    def extract_issue_details(self, issue):
        issue_dict = {
            "issue_number": issue["number"],
            "issue_title": issue["title"],
            "issue_body": issue["body"],
            "issue_url": issue["html_url"],
            "issue_state": issue["state"],
            "creator": issue["user"]["login"],
            "labels": (
                ", ".join([label["name"] for label in issue["labels"]])
                if issue["labels"]
                else ""
            ),
            "owner": self.extract_assignees(issue),
        }
        issue_dict["component"] = self.component_split(issue_dict["labels"])

        return issue_dict

    def extract_assignees(self, issue, contribution_type="last"):
        assignees = issue["assignees"]
        if assignees:
            return ",".join([assignee["login"] for assignee in assignees])

        timeline = issue["timeline_data"]
        last_assignment = None
        contributors = []

        for timeline_event in timeline:
            event = timeline_event["event"]
            if event == "cross-referenced" and timeline_event["source"]["issue"].get(
                "pull_request", None
            ):
                last_assignment = timeline_event.get("actor")

                if last_assignment:
                    last_assignment = last_assignment.get("login")
                    contributors.append(last_assignment)

            if event == "referenced" and timeline_event["commit_url"]:
                last_assignment = timeline_event.get("actor")

                if last_assignment:
                    last_assignment = last_assignment.get("login")
                    contributors.append(last_assignment)

        most_contributed = None
        if len(contributors) > 0:
            most_contributed = Counter(contributors).most_common(1)[0][0]

        if contribution_type == "last":
            return last_assignment
        elif contribution_type == "most":
            return most_contributed
        else:
            raise Exception("Unsupported contribution type.")

    def component_split(self, labels):
        components = []
        for label in labels.split(","):
            if "comp:" in label.lower():
                components.append(label.strip())

        return ",".join(components)


json_root = "/work/disa_lab/afif/projects/openj9/issue_repository"
extractor = IssueExtractor()
issues = extractor.extract_issues(json_root=json_root)

df = pd.DataFrame(issues)
df = df.sort_values(by="issue_number")
df = df.assign(owner=df["owner"].str.split(","))
df = df.explode("owner")
df["owner"] = df["owner"].str.lower()
df = df.assign(component=df["component"].str.split(","))
df = df.explode("component")

print("Component summary:\n")
print(df.component.value_counts())
df.to_csv("openj9_22112024_explode.csv", index=False)
