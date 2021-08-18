# A PyTorch Semi-Supervised Active Learning Framework for Object Detection
<img src="figures/pytorch-logo-dark.png" width="15%">

This is a PyTorch implementation for localization-oriented active learning appraoches in semi-supervised setting (in specific, pseudolabels). 

**For personal practice only; current experiments solely on medical image datasets.**

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [A PyTorch Semi-Supervised Active Learning Framework for Object Detection](#a-pytorch-semi-supervised-active-learning-framework-for-object-detection)
  - [Requirements](#requirements)
  - [Script Structure](#script-structure)
  - [Usage](#usage)
    - [Config file format](#config-file-format)
    - [Using config files](#using-config-files)
    - [Resuming from checkpoints](#resuming-from-checkpoints)
    - [Using Multiple GPU](#using-multiple-gpu)
  - [Acknowledgements](#acknowledgements)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5 (3.6 recommended)
* PyTorch >= 0.4 (1.2 recommended)
* tqdm (Optional for `test.py`)
* tensorboard >= 1.14 (see [Tensorboard Visualization](#tensorboard-visualization))
* albumentation (latest version from the master branch on GitHub)

Run ``pip install -r requirements.txt`` if needed.

## Script Structure
  ```
  requirements.txt - dependencies
  │
  config.json - configuration for training
  │
  train.py - main script to start training
  │
  test.py - evaluation of trained model
  │
  med_sslal/ - source code
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── config/ - train, test configurations
  │   └── parse_config.py
  │ 
  ├── al/ - all related to active learning
  │   ├── al_criteria.py
  │   └── al_helpers.py
  │
  ├── data/ - datasets, dataloader
  │   ├── dataset.py
  │   ├── sampler.py
  │   └── ... - custom dataset classes
  │
  ├── eval/  - evaluation metrics
  │   ├── metric.py
  │   ├── eval_utils.py
  │   ├── coco_utils.py
  │   ├── coco_eval.py
  │   └── transforms.py
  │
  ├── model/  - define all detection models
  │   ├── model.py
  │   ├── backbone.py
  │   ├── detectors.py
  │   ├── roi_heads.py
  │   └── transform.py
  │
  ├── trainer/ - class which monitors training
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── train_util.py
  ```

## Usage
Try `python train.py -c config.json` to run code.

### Config file format
Config files are in `.json` format:
```javascript
{
    "name": "frcnn_med_alssl",

    "n_gpu": 1,           // number of GPUs used for training
    "seed": 202108,       // random seed for reproducibility

    "n_workers": 2,       // number of cpu processes to be used for data loading
    "batch_size": 4, 

    "arch": {
        "type": "FasterRCNN",
        "args": {
            "backbone_name": "resnet50", 
            "fpn": true,        // whether have fpn on top of backbone
            "pretrained": true,
            "num_classes": 2    // number of object classes (including background)
        }
    },

    "dataset": {
        "type": "DeepLesion",
        "args": {
            "eval_split": 0.2,   // proportion of validation+test 

            "root": "data/DeepLesion",
            "dataset_type": "non-specified", 
            "lesion_type": "lung"
        }
    },

    "optimizer": {          // optimizer from torch.optim     
        "type": "Adam",
        "args":{
            "lr": 3e-4,
            "weight_decay": 1e-8,
            "amsgrad": true
        }
    },

    "metrics": ["sensitivity_k_fps"],   // list of metrics for evaluation

    "lr_scheduler": {       // learning rate scheduler from torch.optim.lr_scheduler
        "type": "CosineAnnealingLR",
        "args": {
            "T_max": 30
        }
    },

    "al_settings": {                    
        "num_cycles": 3,    // number of cycles in active learning
        "type": "ActiveLearningHelper" ,    // initialize a helper class  
        "args": {
            "init_num": 400,    // number of labels in the training set before AL starts
            "budget_num": 200,  // number of labels added per AL cycle
            // criterion used to select samples for labels
            "al_criterion": ["localization_tightness", \
                              "k_means_diversity"]
        }
    },

    "ssl_settings": {
        "include_pseudolabels":true
    },

    "trainer": {
        "epochs": 50,               // number of training epoches per AL cycle
        "save_dir": "exp_results",
        "save_period": 10,          // save checkpoints every save_freq epochs
        "verbosity": 2,             // 0: quiet, 1: per epoch, 2: full
        
        "monitor": "min val_loss",  // mode and metric for model performance monitoring. set 'off' to disable.
        "early_stop": 10,           // number of epochs to wait before early stop. set 0 to disable.

        "tensorboard": true         // enable tensorboard visualization
    }
}
```

Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
  ```

### Acknowlegements
This repo is structured based on [PyTorch Template Library](https://github.com/victoresque/pytorch-template), a great, easy-to-use PyTorch template framework. 

Many implementations regarding active learning are adapted from [Consistency-basd Active Learning for Object Detection](https://github.com/we1pingyu/CALD) by we1pingyu.