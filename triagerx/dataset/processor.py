import pandas as pd
from loguru import logger


class DatasetProcessor:
    @staticmethod
    def load_dataframe(path: str) -> pd.DataFrame:
        logger.debug(f"Loading dataframe: {path}")
        return pd.read_csv(path)

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        logger.debug("Cleaning dataset...")
        df["text"] = df["text"].str.replace(
            "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            " ",
        )
        df["text"] = df["text"].str.replace(" +", " ", regex=True)

        return df

    @staticmethod
    def prepare_dataframe(df: pd.DataFrame, sample_threshold: int = 0) -> pd.DataFrame:
        logger.debug(
            f"Filtering developers based on minimum contribution: {sample_threshold}..."
        )
        df = df[df["assignees"].notna()]
        developers = df["assignees"].value_counts()
        filtered_developers = developers.index[developers >= sample_threshold]
        df = df[df["assignees"].isin(filtered_developers)]

        logger.debug("Generating 'text' field...")
        df["text"] = df.apply(
            lambda x: "Title: "
            + str(x["issue_title"])
            + "\nDescription: "
            + str(x["issue_body"]),
            axis=1,
        )

        min_length = 15
        logger.debug(f"Dropping rows with 'text' length < {min_length}...")
        df = df[df["text"].str.len().gt(min_length)]

        df["owner_id"] = pd.factorize(df["assignees"])[0]

        return df

    @staticmethod
    def process_dataset(path: str, sample_threshold: int = 0) -> pd.DataFrame:
        df = DatasetProcessor.load_dataframe(path=path)
        df = DatasetProcessor.prepare_dataframe(
            df=df, sample_threshold=sample_threshold
        )
        df = DatasetProcessor.clean_data(df=df)

        return df
