import json
import os
import re

import pandas as pd

df = pd.read_csv(
    "/home/mdafifal.mamun/notebooks/triagerX/app/app_data/openj9_train_17.csv"
)

monitored_owners = list(df["owner"].unique())
print(f"Total unique owners: {len(monitored_owners)}", monitored_owners)


def read_json_file(file_path):
    """
    Read a JSON file and return its content.
    """
    with open(file_path, "r") as file:
        return json.load(file)


def get_participants(issue):
    """
    Extract participants from a single issue dictionary.
    Returns a dictionary with participant types:
    {
        "direct_assignment": [(login, None)],
        "pull_requests": [(login, created_at)],
        "commits": [(login, created_at)],
        "discussion": [(login, created_at)]
    }
    """
    contributions = {}

    # Direct assignees
    assignees = issue.get("assignees", [])
    assignee_logins = (
        [(assignee["login"], None) for assignee in assignees] if assignees else []
    )
    contributions["direct_assignment"] = assignee_logins

    # Timeline analysis
    timeline = issue.get("timeline_data", [])
    pull_requests, commits, discussion = [], [], []

    for timeline_event in timeline:
        event = timeline_event.get("event")
        created_at = timeline_event.get("created_at")
        actor = timeline_event.get("actor", {})

        if not actor:
            continue

        actor_login = actor.get("login")

        if event == "cross-referenced" and timeline_event.get("source", {}).get(
            "issue", {}
        ).get("pull_request"):
            pull_requests.append((actor_login, created_at))
        elif event == "referenced" and timeline_event.get("commit_url"):
            commits.append((actor_login, created_at))
        elif event == "commented":
            discussion.append((actor_login, created_at))

    contributions["pull_requests"] = pull_requests
    contributions["commits"] = commits
    contributions["discussion"] = discussion

    return contributions


def get_developer_predictions(json_file):
    timeline_data = json_file.get("timeline_data", [])

    for data in timeline_data:
        body = data.get("body")
        if body and "Recommended Assignees" in body:
            # Extract the developer predictions using regex
            match = re.search(r"Recommended Assignees:\s*(.*)\n", body)
            match2 = re.search(r"Recommended Components:\s*(.*)\n", body)
            developers = []
            components = []
            if match:
                developers = match.group(1).strip().split(",")
                developers = [dev.strip().lower() for dev in developers if dev.strip()]

            if match2:
                components = match2.group(1).strip().split(",")
                components = [
                    comp.strip().lower() for comp in components if comp.strip()
                ]

            return developers, components
    return [], []


all_preds = os.listdir("./test_issues_dump")
all_preds = [
    file_name
    for file_name in all_preds
    if int(file_name.split(".")[0]) > 20310 and int(file_name.split(".")[0]) < 25000
]
actual_contributions = 0
actual_contributions_with_discussions = 0
total_correct = 0
total_correct_with_discussions = 0

for pred in all_preds:
    if not pred.endswith(".json"):
        continue
    file_path = os.path.join("./test_issues_dump", pred)
    print("=" * 20)
    try:
        json_data = read_json_file(file_path)
        developers, _ = get_developer_predictions(json_data)
        participants = get_participants(json_data)
        print(f"Developers predicted: {developers}")

        actual_participants = set()
        actual_participants_with_discussions = set()

        # for p in participants["direct_assignment"]:
        #     actual_participants.add(p[0])
        #     actual_participants_with_discussions.add(p[0])
        for p in participants["pull_requests"]:
            actual_participants.add(p[0])
            actual_participants_with_discussions.add(p[0])
        for p in participants["commits"]:
            actual_participants.add(p[0])
            actual_participants_with_discussions.add(p[0])
        for p in participants["discussion"]:
            actual_participants_with_discussions.add(p[0])

        actual_participants = {p.lower() for p in actual_participants}
        actual_participants_with_discussions = {
            p.lower() for p in actual_participants_with_discussions
        }
        print("Participants before filtering:", actual_participants)
        participants = [p for p in actual_participants if p in monitored_owners]
        participants_with_discussions = [
            p for p in actual_participants_with_discussions if p in monitored_owners
        ]

        print(f"Actual commiters: {participants}")
        print(f"Actual participants with discussions: {participants_with_discussions}")

        if len(participants_with_discussions) == 0:
            print(f"No valid participants found in {pred}. Skipping.")
            continue

        if len(participants) > 0:
            actual_contributions += 1

        if len(participants_with_discussions) > 0:
            actual_contributions_with_discussions += 1

        # # Check if at least one monitored owner is in the participants
        # developers = [dev for dev in developers if dev in monitored_owners]
        # if not developers:
        #     print(f"No monitored developers found in {pred}. Skipping.")
        #     exit()

        # if (
        #     len(participants["pull_requests"]) > 0
        #     or len(participants["commits"]) > 0
        #     or len(participants["direct_assignment"]) > 0
        # ):
        #     actual_commits += 1
        for dev in developers:
            if dev in participants_with_discussions:
                print(f"{dev} is a participant in this issue.")
                total_correct_with_discussions += 1
                break

        for dev in developers:
            if dev in participants:
                print(f"{dev} is a participant in this issue.")
                total_correct += 1
                break
                # if dev in [p[0] for p in participants["direct_assignment"]]:
                #     # print(f"{dev} is a direct assignee.")
                #     total_correct += 1
                # elif dev in [p[0] for p in participants["pull_requests"]]:
                #     # print(f"{dev} has made pull requests.")
                #     total_correct += 1
                # elif dev in [p[0] for p in participants["commits"]]:
                #     # print(f"{dev} has made commits.")
                #     total_correct += 1
                # elif dev in [p[0] for p in participants["discussion"]]:
                # print(f"{dev} has participated in discussions.")
                # total_correct += 1
                _ = 1
            else:
                print(f"{dev} has no contributions in this issue.")
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from {pred}: {e}")
    except Exception as e:
        print(f"An error occurred while processing {pred}: {e}")
print("=" * 20)
print("\n\n")
print(
    f"Total correct predictions: {total_correct} out of {actual_contributions}, Acc: {total_correct / actual_contributions}"
)
print(
    f"Total correct predictions with discussion: {total_correct_with_discussions} out of {actual_contributions_with_discussions}, Acc: {total_correct_with_discussions / actual_contributions_with_discussions}"
)
