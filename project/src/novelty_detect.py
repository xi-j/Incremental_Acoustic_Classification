import torch

def make_novelty_detector(mode='confusion'):
    if mode == 'confusion':
        return confusion_novelty_detector


def confusion_novelty_detector(prev_acc, cur_acc, seen_classes, threshold):
    acc_drop = prev_acc - cur_acc
    max_drop_idx = torch.argmax(acc_drop)

    if (acc_drop[max_drop_idx] / prev_acc[max_drop_idx]) >= threshold:
        return False, seen_classes[max_drop_idx]
    else:
        return True, -1
    