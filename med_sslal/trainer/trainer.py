import numpy as np
import sys, math
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, SmoothedValue, MetricLogger, warmup_lr_scheduler, reduce_dict
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, metric_ftns, optimizer, config, device, cycle, 
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):

        super().__init__(model, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        self.cycle = cycle

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        self.lr_scheduler = lr_scheduler
        #self.log_step = 2*data_loader.batch_size
        self.log_step = 50

        self.train_metrics = MetricTracker('loss', 'lr', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Train an epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        task_lr_scheduler = None

        # Warming up
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.data_loader) - 1)

            task_lr_scheduler = warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)

        for batch_idx, (images, targets) in enumerate(tqdm(self.data_loader)):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            _, task_loss_dict = self.model(images, targets)
            
            task_losses = sum(loss for loss in task_loss_dict.values())
            # reduce losses over all GPUs for logging purposes
            task_loss_dict_reduced = reduce_dict(task_loss_dict)
            task_loss_value = sum(loss for loss in task_loss_dict_reduced.values())
            
            if not math.isfinite(task_loss_value):
                print("Loss is {}, stopping training".format(task_loss_value))
                print(task_loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            task_losses.backward()
            self.optimizer.step()

            if task_lr_scheduler is not None:
                task_lr_scheduler.step()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', task_loss_value)
            self.train_metrics.update('lr', self.optimizer.param_groups[0]['lr'])

            # DO NOT compute metrics for rcnns, no detections are returned when in train mode
            # for met in self.metric_ftns:
            #     self.train_metrics.update(met.__name__, met(outputs, targets))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Cycle:[{0}] Epoch:[{1}]'.format(self.cycle, epoch)+self._progress(batch_idx), task_loss_value)
                # make grid takes an array of 3d tensors and make it 4d
                #self.writer.add_image('input', make_grid(np.array(images)).cpu(), nrow=8, normalize=True)

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.valid_data_loader):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                _, outputs = self.model(images)
                outputs = [{k: v.to(self.device) for k, v in o.items()} for o in outputs]

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(outputs, targets))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'cycle': self.cycle,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }

        filename = str(self.checkpoint_dir / 'checkpoint-cycle{0}-epoch{1}.pth'.format(self.cycle, epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_cycle{}_best.pth'.format(self.cycle))
            torch.save(state, best_path)
            self.logger.info("Saving cycle{} current best: model_best.pth ...".format(self.cycle))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # check AL settings
        if checkpoint['config']['al_settings'] != self.config['al_settings']:
            self.logger.warning("Warning: Active learning configuration given in config file is different from that of "
                                "checkpoint")

        if checkpoint['cycle'] != self.cycle:
            self.logger.warning("Warning: Active learning is in different stages.") 

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
