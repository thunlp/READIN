def get_bin_metrics(labels: list, preds: list) -> dict:
    '''
    Metrics for binary labels.
    Preds can have values other than 0 and 1.
    ```
                    labels
                  |  0  |  1  
            ----- | --- | --- 
    preds     0   |  A  |  B
              1   |  C  |  D 
            other |  E  |  F

    recall of 0:    A / (A + C + E)
    precision of 0: A / (A + B)
    recall of 1:    D / (B + D + F)
    precision of 1: D / (C + D)
    ```
    '''
    assert len(preds) == len(labels)
    n = len(preds)
    mat = [[0, 0] for _ in range(3)]
    for pred, label in zip(preds, labels):
        if pred not in [0, 1]:
            mat[2][label] += 1 
        else:
            mat[pred][label] += 1
    # print(mat)
    result = {}
    if mat[0][0] + mat[0][1] > 0:
        r0 = mat[0][0] / (mat[0][0] + mat[1][0] + mat[2][0])
        p0 = mat[0][0] / (mat[0][0] + mat[0][1])
        f1_0 = 2 * r0 * p0 / (r0 + p0)
        result.update({
            'recall_0': r0,
            'precision_0': p0,
            'f1_0': f1_0,
        })
    if mat[1][1] + mat[1][0] > 0:
        r1 = mat[1][1] / (mat[0][1] + mat[1][1] + mat[2][1])
        p1 = mat[1][1] / (mat[1][1] + mat[1][0])
        f1_1 = 2 * r1 * p1 / (r1 + p1)
        result.update({
            'recall_1': r1,
            'precision_1': p1,
            'f1_1': f1_1,
        })
    if 'f1_0' in result and 'f1_1' in result:
        result['macro_f1'] = (f1_0 + f1_1) / 2

    acc = (mat[0][0] + mat[1][1]) / n
    result['acc'] = acc

    return result