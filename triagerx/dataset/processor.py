import pandas as pd
from loguru import logger


class Processor:
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
    def prepare_dataframe(
        df: pd.DataFrame, minimum_contribution: int = 0
    ) -> pd.DataFrame:
        logger.debug(
            f"Filtering developers based on minimum contribution: {minimum_contribution}..."
        )
        developers = df["assignees"].value_counts()
        filtered_developers = developers.index[developers >= minimum_contribution]
        df = df[df["assignees"].isin(filtered_developers)]

        logger.debug("Generating 'text' field...")
        df["text"] = df.apply(
            lambda x: str(x["issue_title"]) + "\n" + str(x["issue_body"]), axis=1
        )
        df["owner_id"] = pd.factorize(df["assignees"])[0]

        min_length = 15
        logger.debug(f"Dropping rows with 'text' length < {min_length}...")
        df = df[df["text"].str.len().gt(min_length)]

        return df

    @staticmethod
    def process_dataset(path: str) -> pd.DataFrame:
        df = Processor.load_dataframe(path=path)
        df = Processor.prepare_dataframe(df=df)
        df = Processor.clean_data(df=df)

        return df
