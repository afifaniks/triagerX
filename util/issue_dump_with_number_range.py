import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


def get_issue(owner, repo, issue_number, headers):
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    issue_url = f"{base_url}/issues/{issue_number}"
    issue_response = requests.get(issue_url, headers=headers)
    if issue_response.status_code == 404:
        return None
    issue_response.raise_for_status()
    return issue_response.json()


def filter_pull_requests(issue):
    return "/pull/" not in issue["html_url"]


def get_field_data(url, headers):
    comments_data = []
    page = 1

    while True:
        comments_response = requests.get(f"{url}?page={page}", headers=headers)
        comments_response.raise_for_status()
        data = comments_response.json()
        comments_data.extend(data)
        if len(data) < 30:
            break
        page += 1

    return comments_data


def dump_issue_data(issue, file_path):
    with open(os.path.join(file_path, f"{issue['number']}.json"), "w") as file:
        file.write(json.dumps(issue, indent=2))


def dump_issues(owner, repo, gh_token, path, min_issue, max_issue):
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {gh_token}",
    }

    for issue_number in range(min_issue, max_issue + 1):
        issue = get_issue(owner, repo, issue_number, headers)
        if not issue:
            print(f"Issue {issue_number} not found. Skipping...")
            continue

        if not filter_pull_requests(issue):
            print(f"Issue {issue_number} is a pull request. Skipping...")
            continue

        destination = os.path.join(path, f"{issue_number}.json")

        if os.path.exists(destination):
            print(f"Issue {issue_number} already downloaded! Skipping...")
            continue

        print(f"Extracting issue data: {issue_number}")
        issue["timeline_data"] = get_field_data(issue["timeline_url"], headers)
        issue["comments_data"] = get_field_data(issue["comments_url"], headers)
        dump_issue_data(issue, path)
        time.sleep(1.2)


gh_token = os.environ["GH_TOKEN_2"]

parser = argparse.ArgumentParser(description="Dump GitHub issues to a specified path.")
parser.add_argument("--path", type=str, required=True, help="Output path for dumping issues")
parser.add_argument("--owner", type=str, required=True, help="Github repo owner")
parser.add_argument("--repo", type=str, required=True, help="Github repo")
parser.add_argument("--min_issue", type=int, required=True, help="Minimum issue number")
parser.add_argument("--max_issue", type=int, required=True, help="Maximum issue number")

args = parser.parse_args()

os.makedirs(args.path, exist_ok=True)

print(f"Dumping data for: {args.owner}/{args.repo}")
dump_issues(args.owner, args.repo, gh_token, args.path, args.min_issue, args.max_issue)
