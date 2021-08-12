from torch.utils.data import DataLoader
from data.sampler import SubsetSequentialSampler
from utils import collate_fn
import torch
from tqdm import tqdm
import numpy as np

from . import al_criteria

__all_unc_supported__ = ['prediction_uncertainty', 'localization_tightness']
__all_rep_supported__ = ['representativeness']
__all_div_supported__ = ['k_means_diversity']
__all_criteria_supported__ = __all_unc_supported__ + __all_rep_supported__ + __all_div_supported__

class ActiveLearningHelper:
    """
    Helper class for performing active learning.
    """
    def __init__(self, init_num, budget_num, num_workers=2, 
                 al_criterion=['localization_tightness', 'k_means_diversity']):
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


    def update(self, model, train_dataset, labeled_set, unlabeled_set, device):
        """
        Run model inference on all unlabeled samples.
        Assign ground truth labels to the most valuable samples, 
        based on the specified AL criterion/criteria.
        """
        labeled_dataloader, unlabeled_dataloader = self._get_dataloaders(train_dataset, labeled_set, unlabeled_set)

        uncertainties, features = [], []
        model.eval()
        with torch.no_grad():
            for images, _ in tqdm(labeled_dataloader):
                images = list(image.to(device) for image in images)
                torch.cuda.synchronize()

                for image in images: 
                    feature, output = model([image]) 
                    uncertainties.append(self.unc_criterion(output))
                    features.append(al_criteria.feature_pooling(feature))
        
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

        return labeled_set, unlabeled_set
                    
        
    def _split_labeled_unlabeled(self, n_samples):
        indices = list(range(n_samples))
        labeled_set = indices[:self.init_num]
        unlabeled_set = indices[self.init_num:]
        return labeled_set, unlabeled_set
    
    def _get_dataloaders(self, train_dataset, labeled_set, unlabeled_set):
        labeled_dataloader = DataLoader(train_dataset, batch_size=1, sampler=SubsetSequentialSampler(labeled_set), 
                                num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)
        
        unlabeled_dataloader = DataLoader(train_dataset, batch_size=1, sampler=SubsetSequentialSampler(unlabeled_set),
                                  num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

        return labeled_dataloader, unlabeled_dataloader


    



        


