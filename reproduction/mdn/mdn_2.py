import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    text = str(text)  # In case, there is nan or something else
    cleaned_text = text.strip()

    cleaned_text = re.sub(r"(https?|ftp):\/\/[^\s/$.?#].[^\s]*", "", cleaned_text)
    cleaned_text = re.sub(r"0x[\da-fA-F]+", "<hex>", cleaned_text)
    cleaned_text = re.sub(r"\b[0-9a-fA-F]{16}\b", "<hex>", cleaned_text)
    cleaned_text = re.sub(
        r"\b\d{2}:\d{2}:\d{2}:\d{4,} GMT\b",
        "",
        cleaned_text,
    )
    cleaned_text = re.sub(
        r"\b\d{2}:\d{2}:\d{2}(\.\d{2,3})?\b",
        "",
        cleaned_text,
    )
    cleaned_text = re.sub(
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\b",
        "",
        cleaned_text,
    )
    cleaned_text = re.sub(r"```", "", cleaned_text)
    cleaned_text = re.sub(r"-{3,}", "", cleaned_text)
    cleaned_text = re.sub(r"[\*#=+\-]{3,}", "", cleaned_text)

    cleaned_text = re.sub(r"(\r?\n)+", "\n", cleaned_text)
    cleaned_text = re.sub(r"(?![\r\n])\s+", " ", cleaned_text)
    cleaned_text = cleaned_text.strip()

    tokens = word_tokenize(cleaned_text)
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]

    return tokens


max_seq_length = 64
dataset_path = "/home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv"

df = pd.read_csv(dataset_path)
df = df.rename(columns={"assignees": "owner", "issue_body": "description"})

df["text"] = df.apply(lambda row: f"{row['issue_title']}\n{row['description']}", axis=1)

df = df.sort_values(by="issue_number")
df = df[df["owner"].notna()]
test_size = 0.1

df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)

sample_threshold = 20
developers = df_train["owner"].value_counts()
filtered_developers = developers.index[developers >= sample_threshold]
df_train = df_train[df_train["owner"].isin(filtered_developers)]

test_owners = set(df_test["owner"])
unwanted = list(test_owners - set(df_train["owner"]))
df_test = df_test[~df_test["owner"].isin(unwanted)]

all_bug_reports = [
    preprocess_text(text)[:max_seq_length] for text in df_train.text.tolist()
]

all_reports_counts = [Counter(report) for report in all_bug_reports]

vocabulary = set(word for report in all_bug_reports for word in report)


def precompute_collection_probabilities(all_reports_counts):
    collection_word_counts = Counter()
    for report_counts in all_reports_counts:
        collection_word_counts.update(report_counts)

    total_words = sum(collection_word_counts.values())
    return {word: count / total_words for word, count in collection_word_counts.items()}


collection_probabilities = precompute_collection_probabilities(all_reports_counts)


def compute_smoothed_unigram_probs(report_counts, collection_probs, mu=0.1):
    total_words = sum(report_counts.values())
    return {
        word: (1 - mu)
        * (report_counts.get(word, 0) / total_words if total_words > 0 else 0)
        + mu * collection_probs.get(word, 0)
        for word in vocabulary
    }


def kl_divergence(query_probs, report_probs):
    return sum(
        query_probs[word] * np.log(query_probs[word] / (report_probs.get(word, 1e-10)))
        for word in query_probs
        if query_probs[word] > 0
    )


def get_kl_scores(query_report):
    query_report_counts = Counter(query_report)
    query_probs = compute_smoothed_unigram_probs(
        query_report_counts, collection_probabilities
    )
    return [
        kl_divergence(
            query_probs,
            compute_smoothed_unigram_probs(report_counts, collection_probabilities),
        )
        for report_counts in all_reports_counts
    ]


def get_contribution_data(issue_number: int):
    """
    Retrieves the contribution data for a given issue number.

    Args:
        issue_number (int): The issue number.

    Returns:
        Dict[str, List[Tuple[str, str]]]: A dictionary containing contribution data.
    """
    contributions = {}
    issue_file = f"{issue_number}.json"
    last_assignment = None

    with open(
        os.path.join(
            "/home/mdafifal.mamun/notebooks/triagerX/data/openj9/issue_data", issue_file
        ),
        "r",
    ) as file:
        issue = json.load(file)
        assignees = issue.get("assignees", [])
        assignee_logins = (
            [(assignee["login"], None) for assignee in assignees] if assignees else []
        )
        contributions["direct_assignment"] = assignee_logins
        timeline = issue.get("timeline_data", [])
        pull_requests, commits, discussion = [], [], []

        for timeline_event in timeline:
            event = timeline_event.get("event")
            created_at = timeline_event.get("created_at")
            actor = timeline_event.get("actor", {})

            if not actor:
                continue

            actor = actor.get("login")

            if event == "cross-referenced" and timeline_event["source"].get(
                "issue", {}
            ).get("pull_request"):
                pull_requests.append((actor, created_at))
                last_assignment = actor
            elif event == "referenced" and timeline_event.get("commit_url"):
                commits.append((actor, created_at))
                last_assignment = actor
            elif event == "commented":
                discussion.append((actor, created_at))

        contributions["pull_request"] = pull_requests
        contributions["commits"] = commits
        contributions["discussion"] = discussion
        contributions["last_assignment"] = (
            [(last_assignment, None)] if last_assignment else []
        )

    return contributions


def construct_mdn(reports, developers, df_train):
    print("Similar reports:", len(reports))
    network = defaultdict(
        lambda: {"commits": defaultdict(int), "comments": defaultdict(int)}
    )
    for report_id in reports:
        report_data = get_contribution_data(report_id)
        owner = df_train[df_train["issue_number"] == report_id].iloc[0]["owner"]
        for event_type in ["commits", "discussion"]:
            for actor, _ in report_data[event_type]:
                if actor != owner and actor in developers:
                    network[actor][event_type][owner] += 1
    return network


def normalize_edge_weights(network):
    max_commit = max(
        (max(dev["commits"].values(), default=0) for dev in network.values()), default=1
    )
    max_comment = max(
        (max(dev["comments"].values(), default=0) for dev in network.values()),
        default=1,
    )
    for dev in network:
        for target in network[dev]["commits"]:
            network[dev]["commits"][target] /= max_commit
        for target in network[dev]["comments"]:
            network[dev]["comments"][target] /= max_comment
    return network


def calculate_developer_rank(developer, network):
    return sum(network[developer]["commits"].values()) + sum(
        network[developer]["comments"].values()
    )


def rank_developers(network):
    return sorted(
        ((dev, calculate_developer_rank(dev, network)) for dev in network),
        key=lambda x: x[1],
        reverse=True,
    )


def generate_prediction_score(ranks):
    topk = 20

    for k in range(1, topk + 1):
        predicted_devs = []
        for rank in ranks:
            predicted_devs.append([dev.lower() for dev, _ in rank[:k]])

        owners = df_test.owner.tolist()[: len(predicted_devs)]
        topk_accuracy = sum(
            [1 for owner, rank in zip(owners, predicted_devs) if owner in rank]
        ) / len(predicted_devs)
        print(f"Top-{k} accuracy: {topk_accuracy}")


ranks = []
all_test_reports = [
    preprocess_text(text)[:max_seq_length] for text in df_test.text.tolist()
]

total_start_time = time.time()
for bug_id in range(len(df_test)):
    print("Predicting for bug_id:", bug_id)
    start_time = time.time()
    query_report = all_test_reports[bug_id]
    print("Generating KL scores...")
    kl_scores = get_kl_scores(query_report)
    kl_scores = dict(zip(df_train.issue_number.tolist(), kl_scores))
    top_similar_issues = sorted(kl_scores.items(), key=lambda x: x[1])[:20]

    # Check if component is not null
    query_component = df_test.iloc[bug_id]["component"]
    similar_component_issues = None
    if not pd.isnull(query_component):
        similar_component_issues = df_train[df_train["component"] == query_component]

    print(similar_component_issues)

    developer_set = set()
    similar_issue_set = set()

    for issue_number, _ in top_similar_issues:
        contributions = get_contribution_data(issue_number)
        similar_issue_set.add(issue_number)
        for key, value in contributions.items():
            for contributor, _ in value:
                developer_set.add(contributor)

    developer_set2 = set()
    # Iterate top similar issues by similar_component_issues
    if similar_component_issues is not None:
        for issue_number in similar_component_issues.issue_number.tolist():
            similar_issue_set.add(issue_number)
            contributions = get_contribution_data(issue_number)
            for key, value in contributions.items():
                for contributor, _ in value:
                    developer_set2.add(contributor)
    developer_set = developer_set.intersection(developer_set2)

    print("Constructing MDN...")
    MDN = construct_mdn(similar_issue_set, developer_set, df_train)
    print("Normalizing edge weights...")
    normalized_MDN = normalize_edge_weights(MDN)

    print("Ranking developers...")
    ranked_developers = rank_developers(normalized_MDN)
    ranks.append(ranked_developers)
    print("Time taken:", time.time() - start_time)
    print("Current score:")
    generate_prediction_score(ranks)

print("Total time taken:", time.time() - total_start_time)
