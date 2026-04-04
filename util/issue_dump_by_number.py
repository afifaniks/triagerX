import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()


def get_issue_by_number(owner, repo, issue_number, headers):
    """
    Retrieve a single issue by number.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_field_data(url, headers):
    """
    Retrieve paginated data from a given URL (e.g., comments or timeline).
    """
    all_data = []
    page = 1

    while True:
        response = requests.get(f"{url}?page={page}", headers=headers)
        response.raise_for_status()
        data = response.json()
        all_data.extend(data)
        if len(data) < 30:
            break
        page += 1

    return all_data


def dump_issue_data(issue, file_path):
    """
    Dump full issue data to a JSON file.
    """
    with open(os.path.join(file_path, f"{issue['number']}.json"), "w") as f:
        json.dump(issue, f, indent=2)


def dump_single_issue(owner, repo, issue_number, gh_token, path):
    """
    Retrieve and dump a single issue by number.
    Skips pull requests.
    """
    headers = {
        "Accept": "application/vnd.github+json, application/vnd.github.mockingbird-preview+json",
        "Authorization": f"Bearer {gh_token}",
    }

    try:
        print(f"Fetching issue #{issue_number} from {owner}/{repo}")
        issue = get_issue_by_number(owner, repo, issue_number, headers)

        # Skip if it's a pull request
        if "/pull/" in issue["html_url"]:
            print(f"Issue #{issue_number} is a pull request. Skipping.")
            return False

        # Fetch timeline and comments
        print("  Fetching timeline...")
        issue["timeline_data"] = get_field_data(issue["timeline_url"], headers)
        print("  Fetching comments...")
        issue["comments_data"] = get_field_data(issue["comments_url"], headers)

        # Dump to disk
        print(f"  Dumping issue #{issue_number} to {path}")
        dump_issue_data(issue, path)
        time.sleep(1.2)
        return True

    except requests.HTTPError as e:
        print(f"Failed to fetch issue #{issue_number}: {e}")
        return False


if __name__ == "__main__":
    gh_token = os.environ["GH_TOKEN"]

    parser = argparse.ArgumentParser(
        description="Dump multiple GitHub issues by number."
    )
    parser.add_argument(
        "--path", type=str, default="./issues_dump", help="Output directory"
    )
    parser.add_argument(
        "--owner", type=str, default="eclipse-openj9", help="Repository owner"
    )
    parser.add_argument("--repo", type=str, default="openj9", help="Repository name")
    parser.add_argument(
        "--start", type=int, default=20300, help="Start issue number (inclusive)"
    )
    parser.add_argument(
        "--end", type=int, default=20600, help="End issue number (inclusive)"
    )

    args = parser.parse_args()

    os.makedirs(args.path, exist_ok=True)

    for issue_number in range(args.start, args.end + 1):
        success = dump_single_issue(
            args.owner, args.repo, issue_number, gh_token, args.path
        )
        if not success:
            print(f"Skipping issue #{issue_number} due to error or pull request.\n")
