import json
import os
import time

import requests
from dotenv import load_dotenv


load_dotenv()


def save_commit_details(
    owner, repo, commit_sha, github_username, github_token, save_dir
):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"
    response = requests.get(url, auth=(github_username, github_token))
    if response.status_code == 200:
        commit_data = response.json()
        print(f"Saving {commit_sha}...")
        with open(os.path.join(save_dir, f"{commit_sha}.json"), "w") as output:
            json.dump(commit_data, output, indent=2)
    else:
        raise Exception(f"Error fetching commit details: {response.status_code}")


def get_all_commits(owner, repo, github_username, github_token, save_root_dir):
    url = f"https://api.github.com/repos/{owner}/{repo}/commits"
    page = 1
    while True:
        print(f"Extracting page {page}...")
        params = {"page": page, "per_page": 100}  # Pagination
        response = requests.get(
            url, params=params, auth=(github_username, github_token)
        )

        if response.status_code == 200:
            commits = response.json()
            if not commits:  # No more commits
                break

            commits_root = os.path.join(save_root_dir, str(page))
            if not os.path.exists(commits_root):
                os.makedirs(commits_root)

            for commit in commits:
                save_commit_details(
                    owner,
                    repo,
                    commit["sha"],
                    github_username,
                    github_token,
                    commits_root,
                )

                time.sleep(1.3)
            page += 1
        else:
            raise Exception(f"Error fetching commits: {response.status_code}")


# Example usage
owner = "eclipse-openj9"
repo = "openj9"
github_username = os.environ["GH_USERNAME"]
github_token = os.environ["GH_TOKEN"]
commits_root_dir = "D:\\Triager X\\triagerX\\util\\data\\commits"
get_all_commits(owner, repo, github_username, github_token, commits_root_dir)
