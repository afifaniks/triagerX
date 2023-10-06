import os

from dotenv import load_dotenv
from github import Auth, Github


load_dotenv()


auth = Auth.Token(os.environ["GH_KEY"])
github = Github()

repo = github.get_repo(full_name_or_id="eclipse-openj9/openj9")
issues = repo.get_issues(state="closed")

for issue in issues:
    print(list(issue.get_comments()))
    print("")
    print(list(issue.get_labels()))
    print("")
    print(issue.assignees)
    print("")
    print(issue.state)
    break








