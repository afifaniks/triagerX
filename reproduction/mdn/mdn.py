import json
import os
import re
import time
from collections import Counter, defaultdict
from functools import lru_cache

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
print("Max sequence length:", max_seq_length)
dataset_path = "/home/mdafifal.mamun/notebooks/triagerX/data/typescript/ts_bug_data.csv"
issue_path = "/home/mdafifal.mamun/notebooks/triagerX/data/typescript/issue_data"

# dataset_path = "/home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_08122024.csv"
# issue_path = "/home/mdafifal.mamun/notebooks/triagerX/data/openj9/issue_data"

mu = 0.4
df = pd.read_csv(dataset_path)
df = df.rename(columns={"assignees": "owner", "issue_body": "description"})

# %%
df.head()

# %%
df["text"] = df.apply(lambda row: f"{row['issue_title']}\n{row['description']}", axis=1)
df["labels"] = df["labels"].apply(lambda x: str(x).split(",") if x else [])

df = df.sort_values(by="issue_number")
df = df[df["owner"].notna()]
df["owner"] = df["owner"].apply(lambda x: str(x).strip().lower())
test_size = 0.1

num_issues = len(df)
print(f"Total number of issues after processing: {num_issues}")
print("All owners:", df["owner"].unique())

df = df.sort_values(by="issue_number")
df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)

sample_threshold = 20
developers = df_train["owner"].value_counts()
filtered_developers = developers.index[developers >= sample_threshold]
df_train = df_train[df_train["owner"].isin(filtered_developers)]

train_owners = set(df_train["owner"])
test_owners = set(df_test["owner"])

unwanted = list(test_owners - train_owners)

df_test = df_test[~df_test["owner"].isin(unwanted)]

print(f"Training data: {len(df_train)}, Validation data: {len(df_test)}")
print(f"Number of developers: {len(df_train.owner.unique())}")

print(f"Train dataset size: {len(df_train)}")
print(f"Test dataset size: {len(df_test)}")


def compute_term_frequencies(reports):
    """Compute term frequency for each report"""
    term_frequencies = []
    for report in reports:
        term_frequencies.append(Counter(report))
    return term_frequencies


# %%
all_train_bug_reports = [
    preprocess_text(text)[:max_seq_length] for text in df_train.text.tolist()
]


# %%
def create_report_counts(reports):
    """Creates word count dictionaries from tokenized reports."""
    all_reports_counts = []
    for report in reports:
        counts = Counter(report)
        all_reports_counts.append(counts)
    return all_reports_counts


# %%
all_train_reports_counts = create_report_counts(all_train_bug_reports)

# %%
vocabulary = set([word for report in all_train_bug_reports for word in report])


def precompute_collection_probabilities(all_reports_counts, vocabulary):
    """Precomputes and stores collection-wide word probabilities."""
    collection_word_counts = {}
    for report_counts in all_reports_counts:
        for word, count in report_counts.items():
            collection_word_counts[word] = collection_word_counts.get(word, 0) + count

    total_words_in_collection = sum(collection_word_counts.values())

    collection_probabilities = {}
    for word, count in collection_word_counts.items():
        collection_probabilities[word] = (
            count / total_words_in_collection if total_words_in_collection > 0 else 0
        )

    return collection_probabilities


def smoothed_unigram(word, report_k_counts, collection_probabilities, vocabulary, mu):
    """Calculates smoothed unigram probability using precomputed values."""

    report_k_total_words = sum(report_k_counts.values())
    report_k_prob = (
        report_k_counts.get(word, 0) / report_k_total_words
        if report_k_total_words > 0
        else 0
    )

    collection_prob = collection_probabilities.get(
        word, 0
    )  # Retrieve precomputed value

    return (1 - mu) * report_k_prob + mu * collection_prob


def kl_divergence(report_q_probs, report_k_probs):
    """Calculates the KL-divergence between two probability distributions."""
    kl = 0.0
    for word, prob_q in report_q_probs.items():
        prob_k = report_k_probs.get(word, 0)  # Handle words not in report_k
        if prob_q > 0 and prob_k > 0:  # Avoid log(0) errors and division by 0
            kl += prob_q * np.log(prob_q / prob_k)
        elif prob_q > 0:  # only add if prob_q is not 0
            kl += prob_q * np.log(prob_q / 1e-10)  # a small value to avoid log(0)
    return kl


# %%
collection_probabilities = precompute_collection_probabilities(
    all_train_reports_counts, vocabulary
)

# %%
all_test_reports = [
    preprocess_text(text)[:max_seq_length] for text in df_test.text.tolist()
]


report_k_probs_list = []
print("Computing all report probabilities...")
for i, report_k_counts in enumerate(all_train_reports_counts):
    report_k_probs = {}
    for word in vocabulary:
        report_k_probs[word] = smoothed_unigram(
            word, report_k_counts, collection_probabilities, vocabulary, mu
        )
    report_k_probs_list.append(report_k_probs)


# %%
def get_kl_scores(query_report):
    query_report_counts = Counter(query_report)
    kl_scores = []

    query_report_probs = {}
    for word in vocabulary:
        query_report_probs[word] = smoothed_unigram(
            word, query_report_counts, collection_probabilities, vocabulary, mu
        )

        # for i, report_k_counts in enumerate(all_reports_counts):
        #     report_k_probs = {}
        #     for word in vocabulary:
        #         report_k_probs[word] = smoothed_unigram(
        #             word, report_k_counts, collection_probabilities, vocabulary, mu
        #         )

    for report_k_probs in report_k_probs_list:
        kl = kl_divergence(query_report_probs, report_k_probs)
        kl_scores.append(kl)

    return kl_scores


@lru_cache(maxsize=50000)
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
        os.path.join(issue_path, issue_file),
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

            actor = actor.get("login").lower()

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
    """Constructs the Multi-Developer Network (MDN) using commit and discussion data."""

    network = defaultdict(
        lambda: {"commits": defaultdict(int), "comments": defaultdict(int)}
    )

    for report_id in reports:
        report_data = get_contribution_data(report_id)  # Get report data from DataFrame
        owner = df_train[df_train["issue_number"] == report_id].iloc[0][
            "owner"
        ]  # Get the assignee/owner

        for commit_info in report_data["commits"]:
            committer = commit_info[0]
            if (
                committer != owner and committer in developers
            ):  # Only add if the committer is not the owner
                network[committer]["commits"][owner] += 1  # Increment commit count

        for comment_info in report_data["discussion"]:
            commenter = comment_info[0]
            if (
                commenter != owner and commenter in developers
            ):  # Only add if commenter is not the owner
                network[commenter]["comments"][owner] += 1  # Increment comment count

    return network


def normalize_edge_weights(network):
    """Normalizes edge weights (commits and comments) in the MDN."""

    # 1. Calculate Maximum Commit and Comment Counts Globally
    max_commit = 0  # Initialize globally
    max_comment = 0  # Initialize globally

    for dev1, interactions in network.items():
        for dev2, commits in interactions["commits"].items():
            max_commit = max(max_commit, commits)  # Find global max commit
        for dev2, comments in interactions["comments"].items():
            max_comment = max(max_comment, comments)  # Find global max comment

    for dev1, interactions in network.items():
        for dev2 in interactions["commits"]:
            network[dev1]["commits"][dev2] = (
                network[dev1]["commits"][dev2] / max_commit + 0.00001
            )

        for dev2 in interactions["comments"]:
            network[dev1]["comments"][dev2] = (
                network[dev1]["comments"][dev2] / max_comment + 0.00001
            )

    return network


def calculate_developer_rank(developer_name, network):
    """Calculates the ranking score for a developer."""

    interactions = network.get(developer_name, {"commits": {}, "comments": {}})
    rank = 0
    for dev2, commit_weight in interactions["commits"].items():
        rank += commit_weight

    for dev2, comment_weight in interactions["comments"].items():
        rank += comment_weight

    # comment_weight = interactions["comments"].get(dev2, 0)

    return rank


def rank_developers(network):
    """Ranks developers based on their ranking scores."""

    developer_ranks = [(dev, calculate_developer_rank(dev, network)) for dev in network]
    developer_ranks.sort(key=lambda x: x[1], reverse=True)
    return developer_ranks


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


# %%
ranks = []

total_start_time = time.time()

samples = 200
seed_value = 77
print("Number of samples:", samples)

# Set seed
np.random.seed(seed_value)
print("Seed value:", seed_value)

for bug_id in tqdm(
    np.random.choice(range(len(df_test)), samples, replace=False),
    total=samples,
    desc="Predicting",
):
    print("Predicting for bug_id:", bug_id)
    start_time = time.time()
    query_report = all_test_reports[bug_id]

    kl_start_time = time.time()
    kl_scores = get_kl_scores(query_report)
    kl_scores = dict(zip(df_train.issue_number.tolist(), kl_scores))
    top_similar_issues = sorted(kl_scores.items(), key=lambda x: x[1])[:30]
    print("KL Time taken:", time.time() - kl_start_time)

    # Check if component is not null
    # Openj9
    query_component = df_test.iloc[bug_id]["component"]
    similar_component_issues = None
    if not pd.isnull(query_component):
        similar_component_issues = df_train[df_train["component"] == query_component]

    # Typescript
    # query_component = df_test.iloc[bug_id]["labels"]
    # similar_component_issues = None
    # if len(query_component) > 0:
    #     # If any of the labels in the query_component is in the labels of the training data
    #     similar_component_issues = df_train[
    #         df_train["labels"].apply(lambda x: bool(set(x) & set(query_component)))
    #     ]
    #     # Select last 100 issues from dataframe
    #     similar_component_issues = similar_component_issues[-20:]

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

    MDN = construct_mdn(similar_issue_set, developer_set, df_train)
    normalized_MDN = normalize_edge_weights(MDN)

    ranked_developers = rank_developers(normalized_MDN)
    ranks.append(ranked_developers)
    print("Time taken:", time.time() - start_time)
    print("Current score:")
    generate_prediction_score(ranks)

print("Total time taken:", time.time() - total_start_time)
