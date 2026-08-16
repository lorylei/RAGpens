def post_process(results):
    processed_results = []
    for text in results:
        prefix = "{'title': "
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
        if "}" in text:
            text = text.split("}", 1)[0]
        text = text.strip()
        if ((text.startswith("'") and text.endswith("'")) or
                (text.startswith('"') and text.endswith('"'))):
            text = text[1:-1]
        processed_results.append(text.strip())
    return processed_results
