import re

def detect_module(text):
    text_lower = text.lower()

    parts = {
        "faq": None,
        "recommendation": None
    }

    if "suggest" in text_lower or "recommend" in text_lower:
        split_words = re.split(r'suggest|recommend', text, flags=re.IGNORECASE)
        parts["faq"] = split_words[0].strip()
        parts["recommendation"] = "suggest " + split_words[1].strip()
    else:
        parts["faq"] = text.strip()

    return parts
