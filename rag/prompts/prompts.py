from prompts.pre_process import get_query


def get_prompt(inp, profile, max_length, tokenizer):
    query = get_query(inp)
    if not profile:
        return (
            "Generate a concise English title for the following article. "
            "Return ONLY a JSON object in this format: {'title': '<your title>'} "
            "(no explanation, no other text).\n\n"
            f"'text': '{query}'\n"
            "{'title':"
        )

    history = "".join(
        f"'text': '{item['text']}' 'title': '{item['title']}'\n"
        for item in profile
    )
    return (
        "You are an expert headline generator. Below are historical user profiles "
        "(articles and their titles):\n"
        f"{history}"
        "Based on the above, generate a concise English title for the following article. "
        "Return ONLY a JSON object in this format: {'title': '<your title>'} "
        "(no explanation, no other text).\n\n"
        f"'text': '{query}'\n"
        "{'title':"
    )
