import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    lam = np.array(lam, dtype=float)
    k = np.array(k, dtype=int)
    pmf = ((math.exp(-lam)) * (lam ** k)) / math.factorial(k)
    
    cdf = 0
    for i in range(k+1):
        cdf += ((lam**i)*math.exp(-lam))/math.factorial(i)

    return {
        "pmf": float(pmf), "cdf": float(cdf)
    }
    
    