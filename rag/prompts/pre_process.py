import re


PREFIX = re.compile(r"^Generate a headline for the following article:\s*", re.IGNORECASE)


def get_query(input_string):
    return PREFIX.sub("", input_string or "").strip()


def get_corpus(profile, use_date=False):
    if use_date:
        return [
            f"'text': '{item['text']}' 'title': '{item['title']}' 'date': '{item['date']}'"
            for item in profile
        ]
    return [
        f"'text': '{item['text']}' 'title': '{item['title']}'"
        for item in profile
    ]
