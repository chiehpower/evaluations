from typing import List

__all__ = [
    'accuracy'
]


def accuracy(y_true: List, y_pred: List) -> float:
    """
    Calculate accuracy score

    Parameters
    ----------
    y_true : list
        True labels
    y_pred : list
        Predicted labels

    Returns
    -------
    float
        Accuracy score

    Examples
    --------
    >>> evaluations.classification.basic.accuracy([1, 1, 0, 0], [1, 1, 1, 0])
    0.75
    """
    count_true = sum([i == j for i, j in zip(y_true, y_pred)])

    count_total = len(y_true)
    return count_true / count_total
