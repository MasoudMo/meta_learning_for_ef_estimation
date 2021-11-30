import torch
import time
from src.utils import util
from src import builders


import random


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
        self.models = util.to_device(self.models, self.device)

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
        start_epoch = 0
        num_epochs = self.train_config.get('num_epochs', 100)

        self.logger.info(
            'Train for {} epochs starting from epoch {}'.format(
                num_epochs, start_epoch))

        # Start training
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_start = time.time()
            self._train_one_epoch(epoch)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr()[0]

            self.logger.infov(
                '[Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(epoch, lr, train_time, self.loss_meter.avg))

            self.scheduler.step()
            self.loss_meter.reset()

            if epoch % 1 == 0:
                eval_metrics = self._evaluate_once(epoch)
                self.checkpointer.save(epoch, eval_metrics)
                self.logger.info(
                    '[Epoch {}] - {}: {:4f}'.format(
                        epoch, self.eval_standard, eval_metrics[self.eval_standard]))
                self.logger.info(
                    '[Epoch {}] - best {}: {:4f}'.format(
                        epoch, self.eval_standard, self.checkpointer.best_eval_metric))

    def _train_one_epoch(self, epoch):
        util.to_train(self.models)
        dataloaders = builders.dataloader_builder.build_train(
            self.data_config, self.tasks)

        for i, dataloader in enumerate(dataloaders):
            context_dataloader, target_dataloader = dataloader['context'], dataloader['target']
            for j, (context_input, context_label), (target_input, target_label) in enumerate(
                zip(context_dataloader, target_dataloader)):
                context_input, context_label =\
                    context_input.to(self.device), context_label.to(self.device)
                target_input, target_label =\
                    target_input.to(self.deivce), target_label.to(self.device)

                output = self.models['np'](
                    self.models['encoder'](context_input), context_label.unsqueeze(0),
                    self.models['encdoer'](target_input), target_label.unsqueeze(0))

                # Compute the NPML objective
                loss = self.criterion(output, target_label)

                # Back propagate
                loss.backward()

                # Do one optimization step
                self.optimizer.step()

                self.logger.info('[Epoch {} Loader {}/{}] NPML train loss: {}'.format(
                    epoch, i, len(dataloaders), loss.detach().item()))

        torch.cuda.empty_cache()
        return

    def _evaluate_once(self, epoch):
        util.to_eval(self.models)
        dataloader = builders.dataloader_builder.build_test(
            self.data_config, self.tasks)
        context_dataloader, target_dataloader =\
            dataloader['context'], dataloader['test']

        for (context_input, context_label), (target_input, target_label) in zip(
            context_dataloader, target_dataloader):

            # Run the LNP model
            output = self.models['np'](
                self.models['encoder'](context_input), context_label.unsqueeze(0),
                self.models['encdoer'](target_input), target_label.unsqueeze(0))

            # Compute the NPML objective
            loss = self.criterion(output, target_label)

            self.logger.info(
                '[Epoch {}] - NPML test loss: {:4f}'.format(
                    epoch, loss.detach().item()))

        return loss

