import os
import json
import sys

import pandas as pd
from tqdm import tqdm


class IssueExtractor:
    def extract_issues(self, json_root):
        all_issues = []
        file_paths = os.listdir(json_root)
        progress_bar = tqdm(total=len(file_paths), desc="Processing issues", unit="files", leave=False)
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
            "labels": ", ".join([label["name"] for label in issue["labels"]]) if issue["labels"] else "",
            "assignees": self.extract_assignees(issue)
        }
        issue_dict["component"] = self.component_split(issue_dict["labels"])

        return issue_dict

    def extract_assignees(self, issue):
        assignees = issue["assignees"]
        if assignees:
            return ",".join([assignee["login"] for assignee in assignees])
        else:
            timeline = issue["timeline_data"]
            for timeline_event in timeline:
                event = timeline_event["event"]
                if event == "cross-referenced" and timeline_event["source"]["issue"].get("pull_request", None):
                    return timeline_event["actor"]["login"]
                if event == "referenced" and timeline_event["commit_url"]:
                    return timeline_event["actor"]["login"]
        return None

    def component_split(self, labels):
        for label in labels.split(","):
            if "comp:" in label.lower():
                return label.strip()
        return None


json_root = "data/issue_data"
extractor = IssueExtractor()
issues = extractor.extract_issues(json_root=json_root)

df = pd.DataFrame(issues)
df = df.sort_values(by="issue_number")
df.to_csv("test.csv", index=False)
