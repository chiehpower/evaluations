from typing import Dict, Tuple, Any
import pandas as pd

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
              'blue': '\033[34m',
              'end': '\033[0m', 
              'bold': '\033[1m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def global_average_precision_score(
        y_true: Dict[Any, Any],
        y_pred: Dict[Any, Tuple[Any, float]]
) -> float:
    indexes = list(y_pred.keys())
    indexes.sort(
        key=lambda x: -y_pred[x][1],
    )
    queries_with_target = len([i for i in y_true.values() if i is not None])
    correct_predictions = 0
    total_score = 0.
    for i, k in enumerate(indexes, 1):
        relevance_of_prediction_i = 0
        if y_true[k] == y_pred[k][0]:
            correct_predictions += 1
            relevance_of_prediction_i = 1
        precision_at_rank_i = correct_predictions / i
        total_score += precision_at_rank_i * relevance_of_prediction_i

    return 1 / queries_with_target * total_score

if __name__=='__main__':

    y_true = {
        'id_001': 123,
        'id_002': None,
        'id_003': 999,
        'id_004': 123,
        'id_005': 999,
        'id_006': 888,
        'id_007': 666,
        'id_008': 666,
        'id_009': None,
        'id_010': 666,
    }
    ytrue = pd.DataFrame(y_true, index=[0]).T
    ytrue.columns = ['Class ID']
    prefix = colorstr('GroundTruth:')
    print("{}:\n{}\n".format(prefix, ytrue))
    
    y_pred = {
        'id_001': (123, 0.15),
        'id_002': (123, 0.10),
        'id_003': (999, 0.30),
        'id_005': (999, 0.40),
        'id_007': (555, 0.60),
        'id_008': (666, 0.70),
        'id_010': (666, 0.99),
    }
    
    ypred = pd.DataFrame(y_pred).T
    ypred.columns = ['Class ID', 'Score']
    prefix = colorstr('Prediction:')
    print(f"{prefix}:\n", ypred)

    gap = global_average_precision_score(y_true, y_pred)
    prefix = colorstr('GAP:')
    print(f"\n{prefix}: ", round(gap, 8))