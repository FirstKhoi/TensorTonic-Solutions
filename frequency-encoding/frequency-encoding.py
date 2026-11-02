def frequency_encoding(values: list) -> list:
    """
    Returns the relative frequency of every input value.
    """
    # Write code here
    n = len(values)
    c = {}
    for v in values:
        c[v] = c.get(v, 0) + 1

    return [c[v] / n for v in values]