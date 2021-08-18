import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..data.sampler import SubsetSequentialSampler
from ..utils import collate_fn
from . import al_criteria

__all_unc_supported__ = ['prediction_uncertainty', 'localization_tightness']
__all_rep_supported__ = ['representativeness']
__all_div_supported__ = ['k_means_diversity']
__all_ssl_supported__ = ['pseudolabel']

class ActiveLearningHelper:
    """
    Helper class for performing active learning.
    """
    def __init__(self, init_num, budget_num, num_workers=2, 
                 al_criterion=['localization_tightness', 'k_means_diversity'],
                 include_pseudolabels=False):
        self.init_num = init_num
        self.budget_num = budget_num
        self.num_workers = num_workers
        
        # validate the specified AL sampling criterion/criteria
        self.criterion_list = al_criterion
        assert len(self.criterion_list) > 0, "No sampling criterion is specified. Please check config settings."
        assert len(self.criterion_list) < 3, "Currently, sampling with more than 2 criteria is unsupported." 
        assert self.criterion_list[0] in __all_unc_supported__, "There must be an uncertainty criterion."
        self.unc_criterion = getattr(al_criteria, self.criterion_list[0])

        # check if a secondary sampling criterion is specified
        self.secondary_criterion = None
        if len(self.criterion_list) == 2:
            assert self.criterion_list[1] in __all_rep_supported__+__all_div_supported__, \
                "The secondary criterion must be a supported diversity or representativeness criterion." 
            self.secondary_criterion = getattr(al_criteria, self.criterion_list[1])
        
        # check if a semi-supervised method is specified
        self.include_pseudolabels = include_pseudolabels


    def update(self, model, train_dataset, labeled_set, unlabeled_set, device):
        """
        Run model inference on all unlabeled samples.
        Assign ground truth labels to the most valuable samples, 
        based on the specified AL criterion/criteria.
        """
        labeled_dataloader, unlabeled_dataloader = self._get_dataloaders(train_dataset, labeled_set, unlabeled_set)

        uncertainties, features, pseudolabels, pseudoflags = [], [], [], []
        model.eval()
        with torch.no_grad():
            for idx, (images, _) in enumerate(tqdm(labeled_dataloader)):
                images = list(image.to(device) for image in images)
                torch.cuda.synchronize()

                for image in images: 
                    feature, output = model([image]) 
                    uncertainties.append(self.unc_criterion(output))
                    features.append(al_criteria.feature_pooling(feature))
                    if self.include_pseudolabels:
                        ps_target = self._get_pseudolabel(output, idx, pseudoflags, threshold=0.8)
                        pseudolabels.append((image.cpu().numpy(), ps_target))
        
        # rank unlabeled samples by their uncertainty
        unc_ranking = np.argsort(uncertainties)
        first_selection_num = self.budget_num * len(self.criterion_list)
        # select the most uncertain samples
        to_be_added = list(np.array(unlabeled_set)[unc_ranking][:first_selection_num])
                
        # if a secondary selection is needed
        if self.secondary_criterion is not None:
            selected_features = np.array(features)[unc_ranking][:first_selection_num]
            to_be_added = self.secondary_criterion(selected_features, self.budget_num, to_be_added)
        
        # update labeled and unlabeled set, respectively
        to_be_added = np.array(to_be_added)
        labeled_set += list(np.array(unlabeled_set)[to_be_added])
        unlabeled_set = list(set(unlabeled_set) - set(to_be_added)) 

        # reorder pseudolabels 
        # since we only pseudolabel those remaining samples w/o groundtruth
        if self.include_pseudolabels:
            pseudo_set = list(set(unc_ranking) - set(to_be_added))
            pseudolabels = [pseudolabels[i] for i in pseudo_set]
            pseudoflags = [pseudoflags[i] for i in pseudo_set]

        return labeled_set, unlabeled_set, pseudolabels, pseudoflags
                    
        
    def _split_labeled_unlabeled(self, n_samples):
        indices = list(range(n_samples))
        labeled_set = indices[:self.init_num]
        unlabeled_set = indices[self.init_num:]
        return labeled_set, unlabeled_set
    
    def _get_dataloaders(self, train_dataset, labeled_set, unlabeled_set):
        labeled_dataloader = DataLoader(train_dataset, batch_size=1, sampler=SubsetSequentialSampler(labeled_set), 
                                num_workers=self.num_workers, collate_fn=collate_fn)
        
        unlabeled_dataloader = DataLoader(train_dataset, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
                                  num_workers=self.num_workers, collate_fn=collate_fn)

        return labeled_dataloader, unlabeled_dataloader
    
    def _get_pseudolabel(self, output, idx, pseudoflags, threshold=0.8):
        target = {}
        
        above_thres = output[0]['scores'].detach().cpu().numpy() >= threshold
        if np.count_nonzero(above_thres) > 0:
            pseudoflags.append(True)
            # only select the top prediction
            boxes = output[0]['boxes'].detach().cpu().numpy()[above_thres][0]
            labels = output[0]['labels'].detach().cpu()[above_thres][0]
            area = (boxes[3] - boxes[1]) * (boxes[2] - boxes[0])
        else:
            pseudoflags.append(False)
            return target

        boxes = torch.as_tensor([boxes], dtype=torch.float32)
        labels = torch.tensor([labels], dtype=torch.int64)
        area = torch.tensor([area])        
        image_id = torch.tensor([idx])
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target



    



        


