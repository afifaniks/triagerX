import re

import pandas as pd
from loguru import logger


class TextProcessor:
    SPECIAL_TOKENS = {
        "hex": "[HEX]",
        "timestamp": "[TIMESTAMP]",
    }

    @staticmethod
    def prepare_dataframe(
        df: pd.DataFrame,
        use_special_tokens: bool,
        use_summary: bool,
        use_description: bool,
        component_training: bool,
        is_openj9: bool = True,
    ) -> pd.DataFrame:
        """
        Prepares the input DataFrame by processing its columns based on the specified options.

        This method modifies the given DataFrame `df` according to the boolean flags provided:
        `use_special_tokens`, `use_summary`, `use_description`, and `component_training`.
        These flags determine whether special tokens, summary text, description text, and
        component-specific processing are applied to the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed.
            use_special_tokens (bool): If True, special tokens like [HEX], [NUMERIC] will be added to the text data.
            use_summary (bool): If True, the summary column will be used in the text data.
            use_description (bool): If True, the description column will be used in the text data.
            component_training (bool): If True, the DataFrame will be processed for component-specific training.

        Returns:
            pd.DataFrame: The processed DataFrame with modifications based on the input flags.
        """

        if component_training:
            df = df[df["labels"].notna()]

        if is_openj9:
            df = df[~df["issue_url"].str.contains("/pull/")]
            df["component"] = df["labels"].apply(TextProcessor.component_split)

        df["text"] = df["issue_title"].progress_apply(
            lambda x: "Bug Title: " + str(x),
        )  # type: ignore

        if use_special_tokens:
            logger.info("Adding special tokens...")
            df["description"] = df["description"].progress_apply(
                TextProcessor.process_special_tokens
            )
        else:
            logger.info("Cleaning text...")
            df["description"] = df["description"].progress_apply(
                TextProcessor.clean_text
            )

        if use_summary:
            logger.info("Adding summary...")
            df["summary"] = df["summary"].progress_apply(TextProcessor.clean_summary)
            df["text"] = df.progress_apply(
                lambda x: x["text"] + "\nBug Summary: " + str(x["summary"]), axis=1
            )  # type: ignore

        if use_description:
            logger.info("Adding description...")
            df["text"] = df.progress_apply(
                lambda x: x["text"] + "\nBug Description: " + str(x["description"]),
                axis=1,
            )  # type: ignore

        min_length = 15
        df = df[df["text"].str.len().gt(min_length)]

        return df

    @staticmethod
    def prepare_text(
        bug_title: str,
        description: str,
    ) -> str:

        processed_text = "Bug Title: " + bug_title
        processed_text += "\nBug Description: " + description

        processed_text = TextProcessor.clean_text(processed_text)

        return processed_text

    @staticmethod
    def process_special_tokens(text: str) -> str:
        text = str(text)  # In case, there is nan or something else
        cleaned_text = text.strip()
        special_tokens = TextProcessor.SPECIAL_TOKENS

        cleaned_text = re.sub(r"(https?|ftp):\/\/[^\s/$.?#].[^\s]*", "", cleaned_text)
        cleaned_text = re.sub(r"0x[\da-fA-F]+", special_tokens["hex"], cleaned_text)
        cleaned_text = re.sub(
            r"\b[0-9a-fA-F]{16}\b", special_tokens["hex"], cleaned_text
        )
        # cleaned_text = re.sub(
        #     r"\b.*/([^/]+)", rf"{special_tokens['filepath']}/\1", cleaned_text
        # )
        # cleaned_text = re.sub(
        #     r"\b([A-Za-z]:)?.*\\([^\\]+)",
        #     rf"{special_tokens['filepath']}/\2",
        #     cleaned_text,
        # )
        # cleaned_text = re.sub(
        #     r"\b(?:\d{1,3}\.){3}\d{1,3}\b", special_tokens["ip"], cleaned_text
        # )
        # cleaned_text = re.sub(
        #     r"(?<!\w)\d+\.\d+\.\d+(\.\d+)*(_\d+)?(-[a-zA-Z]+\d*)?(?!\w)",
        #     special_tokens["version"],
        #     cleaned_text,
        # )
        cleaned_text = re.sub(
            r"\b\d{2}:\d{2}:\d{2}:\d{4,} GMT\b",
            special_tokens["timestamp"],
            cleaned_text,
        )
        cleaned_text = re.sub(
            r"\b\d{2}:\d{2}:\d{2}(\.\d{2,3})?\b",
            special_tokens["timestamp"],
            cleaned_text,
        )
        cleaned_text = re.sub(
            r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\b",
            special_tokens["timestamp"],
            cleaned_text,
        )
        # cleaned_text = re.sub(
        #     r"\b[-+]?\d*\.\d+([eE][-+]?\d+)?\b", special_tokens["numeric"], cleaned_text
        # )
        # cleaned_text = re.sub(r"\d{4,}\b", special_tokens["numeric"], cleaned_text)
        # cleaned_text = re.sub(
        #     r"=\s*-?\d+", f'= {special_tokens["param"]}', cleaned_text
        # )
        cleaned_text = re.sub(r"```", "", cleaned_text)
        cleaned_text = re.sub(r"-{3,}", "", cleaned_text)
        cleaned_text = re.sub(r"[\*#=+\-]{3,}", "", cleaned_text)

        for special_token in special_tokens.values():
            sp_token = special_token[1:-1]
            cleaned_text = re.sub(
                rf"\[{sp_token}\]\s*(\[{sp_token}\]\s*)+",
                f"{special_token}",
                cleaned_text,
            )

        cleaned_text = re.sub(r"(\r?\n)+", "\n", cleaned_text)
        cleaned_text = re.sub(r"(?![\r\n])\s+", " ", cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    @staticmethod
    def component_split(x: str):
        x_split = str(x).split(",")

        for s in x_split:
            if "comp:" in s.lower():
                return s.strip()

        return None

    @staticmethod
    def clean_summary(summary: str) -> str:
        summary = str(summary)
        summary = re.sub("Here is a summar.*?:", "", summary)
        summary = re.sub("\s+", " ", summary)
        summary = summary.strip()

        return summary

    @staticmethod
    def clean_text(text: str) -> str:
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

        return cleaned_text
