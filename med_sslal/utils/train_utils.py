import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random, os
import pickle

from .util import collate_fn

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_random_seed(seed):
     os.environ['PYTHONHASHSEED'] = str(seed)
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.enabled = False 

def save_labeled_unlabeled(config, cycle, labeled_set, unlabeled_set):
    data = {'labeled_set': labeled_set,
            'unlabeled_set': unlabeled_set}
    with open(os.path.join(config.save_dir, 'labeled_unlabeled_cycle{}'.format(cycle)), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_train_data_loader(config, logger, labeled_set, train_dataset, pseudo_dataset):
    logger.info('Current train dataset size: {}'.format(len(train_dataset)))
    if len(pseudo_dataset) >= 0:
        logger.info('{} pseudolabeled samples are added.'.format(len(pseudo_dataset)))
        concat_train_dataset = torch.utils.data.ConcatDataset([train_dataset, pseudo_dataset])
        train_sampler = SubsetRandomSampler(list(set(range(len(train_dataset), len(concat_train_dataset))) | set(labeled_set)))
        train_dataset = concat_train_dataset
        logger.info('Current train dataset size: {}'.format(len(train_dataset)))
    else: 
        train_sampler = SubsetRandomSampler(labeled_set)
    

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, config['batch_size'], drop_last=True)

    # Fixed seed for workers for reproducibility (if num_workers > 2)
    g_seed = torch.Generator()
    g_seed.manual_seed(config['seed'])

    # get train data loader (need to be updated at each AL cycle)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, 
                                                    worker_init_fn=seed_worker, generator=g_seed,
                                                    num_workers=config['n_workers'], collate_fn=collate_fn)

    return train_data_loader