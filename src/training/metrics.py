from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(all_targets, all_preds, epoch):
    if len(all_targets) == 0:
        return 0.0, 0.0, 0.0
    avg = "macro" if epoch > 5 else "micro"
    p = precision_score(all_targets, all_preds, average=avg, zero_division=0)
    r = recall_score(all_targets, all_preds, average=avg, zero_division=0)
    f = f1_score(all_targets, all_preds, average=avg, zero_division=0)
    return p, r, f
