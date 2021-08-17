import torch
from torch import nn
import sklearn
from sklearn.cluster import KMeans
import random

def prediction_uncertainty(output):
    return output[0]['prob_max']

def localization_tightness(output):
    uncertainty = 1.0
    for box, prop, prob_max in zip(output[0]['boxes'], output[0]['props'], output[0]['prob_max']):
        iou = calcu_iou(box, prop)
        u = torch.abs(iou + prob_max - 1)
        uncertainty = min(uncertainty, u.item())
    
    return uncertainty

def k_means_diversity(features, budget_num, indices, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit(features)
    cluster_labels = clusters.labels_
    cluster_indices = indices
    cluster_dict = {}
    
    # group unlabeled indices by cluster
    for label, index in zip(cluster_labels, cluster_indices):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(index)
    
    # iterate over clusters, choose one sample each iteration until meet the budget number
    to_be_added = []
    count = 0
    while count < budget_num:
        for c in range(n_clusters):
            if len(cluster_dict[c]) > 0:
                selected = random.choice(cluster_dict[c])
                to_be_added.append(selected)
                cluster_dict[c].remove(selected)
                count += 1
                if count == budget_num:
                    break
            else:
                continue
    return to_be_added

# Helper functions
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

def feature_pooling(features):
    adaptive_pooling = nn.AdaptiveAvgPool2d(1)
    output = adaptive_pooling(features['pool'])
    output = output.view(output.size(0), -1)

    return output.squeeze(0).detach().cpu().numpy()