def format_input(entity: str, text: str) -> str:
    """Formats a string to structure an input prompt for sentiment analysis about a given entity.

    Args:
        entity (str): The name of the entity for which sentiment is to be analyzed.
        text (str): The context or sentence in which the sentiment about the entity is expressed.

    Returns:
        str: A formatted string with the entity wrapped in <e> tags for emphasis.

    Example:
        >>> format_input("Apple", "I love their new iPhone!")
        'What is the sentiment about [ENTITY] <e>Apple</e> in: I love their new iPhone!'
    """
    return f"What is the sentiment about [ENTITY] <e>{entity}</e> in: {text}"
