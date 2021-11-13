if __name__ == '__main__':
    import torch
    from src.novelty_detect import make_novelty_detector

    novelty_detector = make_novelty_detector()

    prev_acc = torch.tensor([1.0,0.92,0.81,0.87])
    cur_acc = torch.tensor([0.92,0.63,0.36,0.81])

    seen_classes = [6,2,3,1]

    threshold = 0.5

    novel, idx = novelty_detector(prev_acc, cur_acc, seen_classes, threshold)

    print(novel, idx)
