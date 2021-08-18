import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def calcu_iou(A, B):
    """ 
    calculate iou(s) between a reference box A and a predicted bounding box B
    """
    width = min(A[2], B[2]) - max(A[0], B[0]) + 1
    height = min(A[3], B[3]) - max(A[1], B[1]) + 1
    if width <= 0 or height <= 0:
        return 0
        
    Aarea = (A[2] - A[0] + 1) * (A[3] - A[1] + 1)
    Barea = (B[2] - B[0] + 1) * (B[3] - B[1] + 1)
    iner_area = width * height
    
    iou = iner_area / (Aarea + Barea - iner_area)

    return iou

def sensitivity_k_fps(outputs, targets, k=4):
    """
    Compute detection sensitivity with k FPs per image.
    """
    #cpu_device = torch.device("cpu")
    assert len(outputs) == len(targets)
    with torch.no_grad():
        sens = []
        for output, target in zip(outputs, targets):
            #target = [{k: v.to(cpu_device) for k, v in t.items()} for t in target]
            #output = [{k: v.to(cpu_device) for k, v in o.items()} for o in output]

            # omit images with no target
            total_POS = len(target['boxes'])            
            if total_POS == 0:
                continue

            TP, FP = 0, 0
            matches = [False] * total_POS
            for pred_box in output['boxes']:
                # For each prediction, check whether it can match any of the ground truth box
                for idx, ground_truth_box in enumerate(target['boxes']):
                    iou = calcu_iou(pred_box.cpu().numpy(), ground_truth_box.cpu().numpy())
                    # if iou exceeds the threshold 0.5, consider it as a POSITIVE
                    if iou > 0.5:
                        TP += 1
                        matches[idx] = True
                        continue
                    else:
                        FP += 1

                # Check number of false positives
                if FP == k:
                    break
            
            FN = matches.count(False)
            s = TP / (TP + FN)
            sens.append(s)
    
    return sum(sens) / len(sens)
