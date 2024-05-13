import re


class TextProcessor:
    SPECIAL_TOKENS = {
        "hex": "[HEX]",
        "timestamp": "[TIMESTAMP]",
        "numeric": "[NUMERIC]",
        "param": "[PARAM_VALUE]",
        "version": "[VERSION]",
        "ip": "[IP_ADDRESS]",
        "filepath": "[FILE_PATH]",
        "url": "[URL]"
    }

    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text) # In case, there is nan or something else
        cleaned_text = text.strip()
        special_tokens = TextProcessor.SPECIAL_TOKENS
        
        cleaned_text = re.sub(r'(https?|ftp):\/\/[^\s/$.?#].[^\s]*', special_tokens["url"], cleaned_text)
        cleaned_text = re.sub(r'0x[\da-fA-F]+', special_tokens["hex"], cleaned_text)
        cleaned_text = re.sub(r'\b[0-9a-fA-F]{16}\b', special_tokens["hex"], cleaned_text)
        cleaned_text = re.sub(r'\b.*/([^/]+)', rf"{special_tokens['filepath']}/\1", cleaned_text)
        cleaned_text = re.sub(r"\b([A-Za-z]:)?.*\\([^\\]+)", rf"{special_tokens['filepath']}/\2", cleaned_text)
        cleaned_text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', special_tokens["ip"], cleaned_text)
        cleaned_text = re.sub(r"(?<!\w)\d+\.\d+\.\d+(\.\d+)*(_\d+)?(-[a-zA-Z]+\d*)?(?!\w)", special_tokens["version"], cleaned_text)
        cleaned_text = re.sub(r'\b\d{2}:\d{2}:\d{2}:\d{4,} GMT\b', special_tokens["timestamp"], cleaned_text)
        cleaned_text = re.sub(r'\b\d{2}:\d{2}:\d{2}(\.\d{2,3})?\b', special_tokens["timestamp"], cleaned_text)
        cleaned_text = re.sub(r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\b', special_tokens["timestamp"], cleaned_text)
        cleaned_text = re.sub(r'\b[-+]?\d*\.\d+([eE][-+]?\d+)?\b', special_tokens["numeric"], cleaned_text)
        cleaned_text = re.sub(r'\d{4,}\b', special_tokens["numeric"], cleaned_text)
        cleaned_text = re.sub(r'=\s*-?\d+', f'= {special_tokens["param"]}', cleaned_text)
        cleaned_text = re.sub(r'```', "", cleaned_text)
        cleaned_text = re.sub(r'-{3,}', "", cleaned_text)
        cleaned_text = re.sub(r'[\*#=+\-]{3,}', "", cleaned_text)
        
        for special_token in special_tokens.values():
            sp_token = special_token[1:-1]
            cleaned_text = re.sub(rf'\[{sp_token}\]\s*(\[{sp_token}\]\s*)+', f"{special_token}", cleaned_text)
            
        cleaned_text = re.sub(r'(\r?\n)+', "\n", cleaned_text)
        cleaned_text = re.sub(r'(?![\r\n])\s+', " ", cleaned_text)
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