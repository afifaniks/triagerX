import requests
import os
import csv
from dotenv import load_dotenv


load_dotenv()


github_username = os.environ["GH_USERNAME"]
github_token = os.environ["GH_TOKEN"]

owner = "eclipse-openj9"
repo = "openj9"

base_url = f"https://api.github.com/repos/{owner}/{repo}"


def fetch_issues(file_name="github_data.csv"):
    page = 1
    per_page = 50

    with open(file_name, "w"):
        pass

    csv_file = open(file_name, "a")
    csv_writer = csv.writer(csv_file)

    while True:
        print(f"Processing Page: {page}")
        issues_url = f"{base_url}/issues?page={page}&per_page={per_page}"
        issues_response = requests.get(issues_url, auth=(github_username, github_token))
        issues = issues_response.json()

        if len(issues) == 0:
            print("Reached to the end. Exiting...")
            csv_file.close()
            break

        extract_issue_details(csv_writer, issues)

        page += 1


def extract_issue_details(csv_writer, issues):
    csv_writer.writerow(
        [
            "issue_number",
            "issue_url",
            "issue_title",
            "issue_body",
            "issue_state",
            "creator",
            "comments",
            "assignees",
        ]
    )
    for issue in issues:
        issue_number = issue["number"]
        issue_title = issue["title"]
        issue_body = issue["body"]
        issue_url = issue["html_url"]
        issue_state = issue["state"]
        issue_creator = issue["user"]["login"]

        comments_url = f"{base_url}/issues/{issue_number}/comments"
        comments_response = requests.get(
            comments_url, auth=(github_username, github_token)
        )
        comments = comments_response.json()
        comment_bodies = [
            f"<comment><user>{comment['user']['login']}</user><body>{comment['body']}</body></comment>"
            for comment in comments
        ]

        assignees = issue["assignees"]
        assignee_logins = (
            [assignee["login"] for assignee in assignees] if len(assignees) > 0 else []
        )

        csv_writer.writerow(
            [
                issue_number,
                issue_url,
                issue_title,
                issue_body,
                issue_state,
                issue_creator,
                "\n".join(comment_bodies),
                ", ".join(assignee_logins),
            ]
        )


if __name__ == "__main__":
    fetch_issues()
