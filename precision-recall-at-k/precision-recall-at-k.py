def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k=recommended[:k]
    hits=0
    for item in top_k:
        if item in relevant:
            hits+=1
    precision=hits/k
    recall=hits/len(relevant)
    return [precision , recall]
    
    