import os
import time
import requests
import json
from dotenv import load_dotenv

load_dotenv()


def get_issues(owner, repo, page, per_page, headers):
    """
    Retrieve issues from GitHub API for a given repository.

    Args:
        owner (str): Owner of the repository.
        repo (str): Name of the repository.
        page (int): Page number of the issues to retrieve.
        per_page (int): Number of issues per page.
        headers (dict): Headers for the request.

    Returns:
        list: List of issues retrieved.
    """
    base_url = f"https://api.github.com/repos/{owner}/{repo}"
    issues_url = f"{base_url}/issues?state=all&page={page}&per_page={per_page}"
    issues_response = requests.get(issues_url, headers=headers)
    issues_response.raise_for_status()
    return issues_response.json()


def filter_pull_requests(issues):
    """
    Filter out pull requests from the list of issues.

    Args:
        issues (list): List of issues.

    Returns:
        list: List of issues excluding pull requests.
    """
    return [issue for issue in issues if "/pull/" not in issue["html_url"]]


def get_field_data(url, headers):
    """
    Retrieve field from a given URL.

    Args:
        url (str): URL to retrieve from.
        headers (dict): Headers for the request.

    Returns:
        list: List of comments retrieved.
    """
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
    """
    Dump issue data to a JSON file.

    Args:
        issue (dict): Issue data.
        file_path (str): Path to save the JSON file.
    """
    with open(os.path.join(file_path, f"{issue['number']}.json"), "w") as file:
        file.write(json.dumps(issue, indent=2))


def dump_issues(owner, repo, gh_token, path):
    """
    Dump issue data for a repository to JSON files.

    Args:
        owner (str): Owner of the repository.
        repo (str): Name of the repository.
        gh_token (str): GitHub API token.
        path (str): Path to save the JSON files.
    """
    page = 1
    per_page = 100
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {gh_token}",
    }

    while True:
        issues = get_issues(owner, repo, page, per_page, headers)

        if not issues:
            print("Reached the end. Exiting...")
            break

        filtered_issues = filter_pull_requests(issues)

        for issue in filtered_issues:
            print(f"Extracting issue data: {issue['number']}")
            issue["timeline_data"] = get_field_data(issue["timeline_url"], headers)
            issue["comments_data"] = get_field_data(issue["comments_url"], headers)
            dump_issue_data(issue, path)
            time.sleep(1.2)

        page += 1


# Example usage
owner = "eclipse-openj9"
repo = "openj9"

gh_token = os.environ["GH_TOKEN"]
path = "D:\\Triager X\\triagerX\\util\\data\\issue_data"

dump_issues(owner, repo, gh_token, path)
