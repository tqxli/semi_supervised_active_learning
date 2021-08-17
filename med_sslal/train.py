import argparse
import collections
import torch
from torch.utils.data.sampler import SubsetRandomSampler

import data.dataset as module_data
import evals as module_metric
import model.model as module_arch
from al import al_helpers
from parse_config import ConfigParser
from trainer import Trainer

from utils import prepare_device, setup_random_seed, evaluate, save_labeled_unlabeled, get_train_data_loader


def main(config):
    # set up random seed for future reproducibility
    SEED = config['seed']
    setup_random_seed(SEED)

    # get logger
    logger = config.get_logger('train')
    
    # load datasets
    dataset = config.init_obj('dataset', module_data, num_workers=config['n_workers'])
    val_data_loader, test_data_loader = dataset.get_val_test_dataloaders() # no augmentations applied
    train_dataset = dataset.get_train_dataset()

    # prepare an active learning helper, split labeled/unlabeled
    al_helper = config.init_obj('al_settings', al_helpers, 
                                num_workers=config['n_workers'],
                                include_pseudolabels=config['ssl_settings']['include_pseudolabels'])
    labeled_set, unlabeled_set = al_helper._split_labeled_unlabeled(len(train_dataset))
    save_labeled_unlabeled(config, 0, labeled_set, unlabeled_set)

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # prepare evaluation metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # begin AL
    for cycle in range(1, config['al_settings']['num_cycles']+1):
        # update labeled/unlabeled distribution using active learning
        labeled_set, unlabeled_set, \
            pseudolabels, pseudoflags = al_helper.update(model, 
                                                         train_dataset, 
                                                         labeled_set, 
                                                         unlabeled_set, 
                                                         device)
        save_labeled_unlabeled(config, cycle, labeled_set, unlabeled_set)

        # get pseudolabels (if requested)
        pseudo_dataset = module_data.PseudoDataset(pseudo_flags=pseudoflags, pseudolabels=pseudolabels)

        # prepare dataloader for training
        train_data_loader = get_train_data_loader(config=config, 
                                                  labeled_set=labeled_set, 
                                                  train_dataset=train_dataset,
                                                  pseudo_dataset=pseudo_dataset)
        
        # set up optimizer & lr scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

        # start training
        trainer = Trainer(model, 
                          metric_ftns=metrics, 
                          optimizer=optimizer, 
                          cycle=cycle,
                          config=config,
                          device=device,
                          data_loader=train_data_loader,
                          valid_data_loader=val_data_loader,
                          lr_scheduler=lr_scheduler)

        trainer.train()

        # test at the end of each AL cycle
        # FIX: only COCO evaluation now, should enable more
        best_checkpoint_path = str(config.save_dir / 'model_best_cycle{}.pth'.format(cycle))
        model.load_state_dict(torch.load(best_checkpoint_path)['state_dict'])
        evaluator = evaluate(model, test_data_loader, device=device)
        coco_stats = evaluator.coco_eval['bbox'].stats
        logger.info("COCO mAP evaluation statistics on the test set:\n{}".format(coco_stats))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Med_SSLAL_exp')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
