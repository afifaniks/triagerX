import os
import json

import pandas as pd


json_root = "data/issue_data"

file_paths = os.listdir(json_root)

code_contribution_events = ["referenced", "cross-referenced"]

all_issues = []

for file_path in file_paths:
    with open(os.path.join(json_root, file_path), "r") as file:
        issue = json.load(file)

        if "/pull/" in issue["html_url"] or issue["state"] == "open":
            continue

        print(f"Issue: {issue['html_url']}")

        issue_number = issue["number"]
        issue_title = issue["title"]
        issue_body = issue["body"]
        issue_url = issue["html_url"]
        issue_state = issue["state"]
        issue_creator = issue["user"]["login"]
        labels = issue["labels"]
        label_names = ", ".join([label["name"] for label in labels]) if labels else ""
        assignees = issue["assignees"]
        assignee_logins = (
            [assignee["login"] for assignee in assignees] if len(assignees) > 0 else None
        )

        if assignee_logins is None:
            print("Found")
            timeline = issue["timeline_data"]
            for timeline_event in timeline:
                event = timeline_event["event"]

                if event == "cross-referenced" and timeline_event["source"]["issue"].get("pull_request", None):
                    actor = timeline_event["actor"]["login"]
                    assignee_logins = actor

                if event == "referenced" and timeline_event["commit_url"]:
                    actor = timeline_event["actor"]["login"]
                    assignee_logins = actor
        else:
            assignee_logins = ",".join(assignee_logins)

        issue_dict = {
            "issue_number": issue_number,
            "issue_url": issue_url,
            "issue_title": issue_title,
            "issue_body": issue_body,
            "issue_state": issue_state,
            "creator": issue_creator,
            # "comments",
            "assignees": assignee_logins,
            "labels": label_names,
        }

        all_issues.append(issue_dict)

df = pd.DataFrame(all_issues)
df.to_csv("test.csv")
