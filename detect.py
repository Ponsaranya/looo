def detect_module(text):
    t = text.lower()

    # Check if user entered any word starting with 'c' followed by digits
    for w in t.split():
        w_clean = w.strip()
        if w_clean.lower().startswith("c") and w_clean[1:].isdigit():
            return "recommendation"

    # Also detect general recommendation questions
    if "recommend" in t or "suggest" in t:
        return "recommendation"

    # Default → FAQ
    return "faq"
