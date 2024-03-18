import time

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()


def dump_issues(owner, repo, gh_username, gh_token, path):
    page = 1
    per_page = 100
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        'Accept': 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        'Authorization': f'Bearer {gh_token}'
    }

    while True:
        issues_url = f"{base_url}/issues?state=all&page={page}&per_page={per_page}"
        issues_response = requests.get(issues_url, auth=(gh_username, gh_token))
        if issues_response.status_code != 200:
            raise Exception(f"Error occurred: {issues_response.status_code}\n{issues_response}")
        issues = issues_response.json()

        if len(issues) == 0:
            print("Reached to the end. Exiting...")
            break

        for issue in issues:
            if "/pull/" in issue["html_url"]:
                print("Pull request found. Skipping...")
                continue

            print(f"Extracting issue data: {issue['number']}")
            timeline_url = issue['timeline_url']

            comments_url = issue['comments_url']

            timeline_response = requests.get(timeline_url, headers=headers)
            if timeline_response.status_code != 200:
                raise Exception(f"Error occurred: {timeline_response.status_code}\n{timeline_response}")
            timeline_data = timeline_response.json()

            issue["timeline_data"] = timeline_data

            comments_response = requests.get(comments_url, auth=(gh_username, gh_token))
            if comments_response.status_code != 200:
                raise Exception(f"Error occurred: {comments_response.status_code}\n{comments_response}")
            comments_data = comments_response.json()

            issue["comments_data"] = comments_data

            with open(os.path.join(path, f"{page}_{issue['number']}.json"), "w") as file:
                file.write(json.dumps(issue, indent=2))

            time.sleep(1.2)

        page += 1


# Example usage
owner = 'eclipse-openj9'
repo = 'openj9'

gh_token = os.environ["GH_TOKEN"]
gh_username = os.environ["GH_USERNAME"]
path = "D:\\Triager X\\triagerX\\util\\data\\issue_data"


dump_issues(owner, repo, gh_username, gh_token, path)

