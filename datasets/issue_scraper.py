from pathlib import Path

import requests
import os
import csv
import time
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Last: 48


github_username = os.environ["GH_USERNAME"]
github_token = os.environ["GH_TOKEN"]

owner = "eclipse-openj9"
repo = "openj9"

base_url = f"https://api.github.com/repos/{owner}/{repo}"

Path("data").mkdir(parents=True, exist_ok=True)


def fetch_issues(file_name=f"data/openj9_data_{int(time.time())}.csv"):
    page = 1
    per_page = 50

    with open(file_name, "w"):
        pass

    csv_file = open(file_name, "a")
    csv_writer = csv.writer(csv_file)

    while True:
        logger.info(f"Processing Page: {page}")
        issues_url = f"{base_url}/issues?state=all&page={page}&per_page={per_page}"
        issues_response = requests.get(issues_url, auth=(github_username, github_token))
        issues = issues_response.json()

        if len(issues) == 0:
            logger.info("Reached to the end. Exiting...")
            csv_file.close()
            break

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

        extract_issue_details(csv_writer, issues)

        page += 1


def extract_issue_details(csv_writer, issues):
    for issue in issues:
        try:
            parse_issue_detail(csv_writer, issue)
        except Exception as err:
            logger.error(
                f"Error occurred while processing: {issue['html_url']}\nIssue: {issue}\n{err}"
            )


def parse_issue_detail(csv_writer, issue):
    issue_number = issue["number"]
    issue_title = issue["title"]
    issue_body = issue["body"]
    issue_url = issue["html_url"]
    issue_state = issue["state"]
    issue_creator = issue["user"]["login"]
    logger.info(f"Extracting issue: {issue_url}")
    comments_url = f"{base_url}/issues/9217/comments"
    comments_response = requests.get(comments_url, auth=(github_username, github_token))
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
