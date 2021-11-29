import torch
import time
from src.utils import util
from src import builders


from data import CamusEfDataset, EchoNetEfDataset
import torch
import logging
from model import Resnet2Plus1D
from npf.architectures import MLP, merge_flat_input
from npf import LNP
from functools import partial
from npf import NLLLossLNPF
from torch.utils.data import random_split, DataLoader
import random
from math import floor
from data import custom_collate_fn


# Initialize logger
logger_level = logging.INFO
logger = logging.getLogger('meta_learning_for_ef_estimation')
logger.setLevel(logger_level)
ch = logging.StreamHandler()
ch.setLevel(logger_level)
logger.addHandler(ch)

random.seed(0)
torch.manual_seed(0)


class BaseEngine(object):

    def __init__(self, config_path, logger, save_dir):
        # Assign a logger
        self.logger = logger

        # Load configs
        config = util.load_config(config_path)

        self.model_config = config['model']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.data_config = config['data']

        self.eval_standard = self.eval_config['standard']

        # Determine which device to use
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.num_devices = torch.cuda.device_count()

        if device == 'cpu':
            self.logger.warn('GPU is not available')
        else:
            self.logger.warn('{} GPUs are ready to be used'.format(self.num_devices))

        self.save_dir = save_dir

    def run(self):
        pass

    def evaluate(self):
        pass



class Engine(BaseEngine):

    def __init__(self, config_path, logger, save_dir):
        super(Engine, self).__init__(config_path, logger, save_dir)

    def _build(self, mode, init=False):
        # Build tasks
        self.tasks = builders.task_builder.build(
            self.data_config, self.logger)

        # Build a model
        self.models = builders.model_builder.build(
            self.model_config, self.logger)
        self.models = util.to_device(self.model, self.device)

        # Build an optimizer, scheduler and criterion
        self.optimizer = builders.optimizer_builder.build(
            self.train_config['optimizer'], self.models, self.logger)
        self.scheduler = builders.scheduler_builder.build(
            self.train_config, self.optimizer, self.logger)
        self.criterion = builders.criterion_builder.build(
            self.train_config, self.model_config, self.logger)
        self.loss_meter = builders.meter_builder.build(
            self.model_config, self.logger)
        self.evaluator = builders.evaluator_builder.build(
            self.eval_config, self.logger)

        # Build a checkpointer
        self.checkpointer = builders.checkpointer_builder.build(
            self.save_dir, self.logger, self.models, self.optimizer,
            self.scheduler, self.eval_standard, init=init)
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(
            mode=mode, checkpoint_path=checkpoint_path, use_latest=False)

    def run(self):
        self._build(mode='train')
        self._train()


    def _train(self):
        start_epoch, num_steps = 0, 0
        num_epochs = self.train_config.get('num_epochs', 100)

        self.logger.info(
            'Train for {} epochs starting from epoch {}'.format(
                num_epochs, start_epoch))

        # Start training
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_start = time.time()
            num_steps = self._train_one_epoch(epoch, num_steps)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr()[0]

            self.logger.infov(
                '[Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(epoch, lr, train_time, self.loss_meter.avg))

            self.scheduler.step()
            self.loss_meter.reset()

            #if epoch % 1 == 0:
            #    eval_metrics = self._evaluate_once(epoch, num_steps)
            #    self.checkpointer.save(epoch, num_steps, eval_metrics)
            #    self.logger.info(
            #        '[Epoch {}] - {}: {:4f}'.format(
            #            epoch, self.eval_standard, eval_metrics[self.eval_standard]))
            #    self.logger.info(
            #        '[Epoch {}] - best {}: {:4f}'.format(
            #            epoch, self.eval_standard, self.checkpointer.best_eval_metric))

    def _train_one_epoch(self, epoch, num_steps):
        util.to_train(self.models)
        dataloaders = builders.dataloader_builder.build(self.tasks)
        context_dataloader = dataloaders['context']
        target_dataloader = dataloaders['target']

        for (context_input, context_label), (target_input, target_label) in zip(
            context_dataloader, target_dataloader):
            context_input, context_label =\
                context_input.to(self.device), context_label.to(self.device)
            target_input, target_label =\
                target_input.to(self.deivce), target_label.to(self.device)

            output = self.models['np'](
                self.models['encoder'](context_input), context_label,
                self.models['encdoer'](target_input), target_label)

            # Compute the NPML objective
            loss = self.criterion(output, target_label)

            # Back propagate
            loss.backward()

            # Do one optimization step
            self.optimizer.step()

            self.logger.info('NPML: {}'.format(loss.detach().item()))

        torch.cuda.empty_cache()


    #def _evaluate_once(self, epoch, num_steps):
    #    dataloader = self.dataloaders['val']
    #    num_batches = len(dataloader)

    #    self.model.eval()
    #    self.logger.info('[Epoch {}] Evaluating...'.format(epoch))

    #    dataloaders = builders.dataloader_builder.build(self.tasks)
    #    target_dataloader = dataloaders['target']


    #    for i, input_dict in enumerate(dataloader):
    #        with torch.no_grad():
    #            input_dict = util.to_device(input_dict, self.device)

    #            # Forward propagation
    #            output_dict = self.models['model'](input_dict)
    #            output_dict['labels'] = input_dict['labels']

    #            # Print losses
    #            self.evaluator.update(output_dict)

    #            # Accumulate precision and recall
    #            self.pr_meter.accumulate(
    #                output_dict['labels'], output_dict['logits'])

    #            #self.logger.info('[Epoch {}] Evaluation batch {}/{}'.format(
    #            #    epoch, i+1, num_batches))
    #            #if epoch == 49 and i < 10:
    #            #    criterion_name = self.train_config['criterion']['name']
    #            #    if criterion_name == 'l2':
    #            #        preds = output_dict['logits']
    #            #    elif criterion_name == 'cross_entropy' or criterion_name == 'softmax_l2':
    #            #        preds = torch.nn.functional.softmax(output_dict['logits'], dim=1)
    #            #    self.writer.add_histogram(tag='prediction results {}'.format(i), values=preds[:1], global_step=num_steps)


    #    self.evaluator.print_log(epoch, num_steps)
    #    torch.cuda.empty_cache()
    #    eval_metric = self.evaluator.compute()

    #    # Add precision and recall curves for each class
    #    labels, preds = self.pr_meter.get_labels_and_preds()
    #    for cls_idx, (label, pred) in enumerate(zip(labels, preds)):
    #        self.writer.add_pr_curve(
    #            'pr {} class'.format(i), label, pred, global_step=num_steps)

    #    # Reset the evaluator
    #    self.evaluator.reset()
    #    return {self.eval_standard: eval_metric}



