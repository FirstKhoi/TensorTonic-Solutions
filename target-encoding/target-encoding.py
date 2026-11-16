def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    # Write code here
    sum = {}
    counts = {}
    for cat, t in zip(categories, targets):
        sum[cat] = sum.get(cat, 0.0) + t
        counts[cat] = counts.get(cat, 0) + 1
    means = {cat: sum[cat] / counts[cat] for cat in sum}
    return [means[cat] for cat in categories]