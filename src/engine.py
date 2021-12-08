import torch
import time
from src.utils import util
from src.builders import model_builder, task_builder, dataloader_builder,\
    optimizer_builder, criterion_builder, checkpointer_builder, evaluator_builder,\
    meter_builder, scheduler_builder
import wandb


import random


random.seed(0)
torch.manual_seed(0)


class BaseEngine(object):

    def __init__(self, config_path, logger, save_dir):
        # Assign a logger
        self.logger = logger

        # Load configs
        config = util.load_config(config_path)

        # Initialize wandb
        wandb.init(project='Meta Learning for EF Estimation',
                   config=config)

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
            self.logger.warn('{} GPU/s are ready to be used'.format(self.num_devices))

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
        self.tasks = task_builder.build(
            self.data_config, self.logger, self.device)

        # Build a model
        self.models = model_builder.build(
            self.model_config, self.logger)
        self.models = util.to_device(self.models, self.device)

        # Build an optimizer, scheduler and criterion
        self.optimizer = optimizer_builder.build(
            self.train_config['optimizer'], self.models, self.logger)
        self.scheduler = scheduler_builder.build(
            self.train_config, self.optimizer, self.logger)
        self.criterion = criterion_builder.build(
            self.train_config, self.model_config, self.logger)
        self.loss_meter = meter_builder.build(
            self.model_config, self.logger)
        self.evaluators = evaluator_builder.build(
            self.eval_config, self.logger, self.device)

        # Build a checkpointer
        self.checkpointer = checkpointer_builder.build(
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

            # Evaluate every 10 epochs
            if epoch % 10 == 0:
                eval_metrics = self._evaluate_once(epoch)
                self.checkpointer.save(epoch, eval_metrics)
                self.logger.info(
                    '[Epoch {}] - {}: {:4f}'.format(
                        epoch, self.eval_standard, eval_metrics[self.eval_standard]))
                self.logger.info(
                    '[Epoch {}] - best {}: {:4f}'.format(
                        epoch, self.eval_standard, self.checkpointer.best_eval_metric))

            wandb.log({'train_npml_loss': self.loss_meter.avg,
                       'val_mae_loss': eval_metrics['mae'],
                       'val_r2_score': eval_metrics['r2']})

            self.scheduler.step()
            self.loss_meter.reset()

    def _train_one_epoch(self, epoch):
        util.to_train(self.models)
        dataloaders = dataloader_builder.build_train(
            self.data_config, self.tasks, self.logger)

        for i, dataloader in enumerate(dataloaders):
            context_inputs, context_labels, target_inputs, target_labels = [], [], [], []
            context_dataloader, target_dataloader = dataloader['context'], dataloader['target']

            for j, ((context_input, context_label), (target_input, target_label)) in enumerate(
                zip(context_dataloader, target_dataloader)):
                context_input, context_label =\
                    context_input.to(self.device), context_label.to(self.device)
                target_input, target_label =\
                    target_input.to(self.device), target_label.to(self.device)

                context_inputs.append(self.models['x_encoder'](context_input))
                context_labels.append(context_label)
                target_inputs.append(self.models['x_encoder'](target_input))
                target_labels.append(target_label)

            context_inputs = torch.stack(context_inputs)
            context_labels = torch.stack(context_labels).unsqueeze(0).permute(0, 2, 1)
            target_inputs = torch.stack(target_inputs)
            target_labels = torch.stack(target_labels).unsqueeze(0).permute(0, 2, 1)
            output = self.models['np'](
                context_inputs, context_labels, target_inputs, target_labels)

            # Compute the NPML objective
            loss = self.criterion(output, target_label)
            self.loss_meter.update(loss.detach().item())

            # Back propagate
            loss.backward()

            # Do one optimization step
            self.optimizer.step()

            self.logger.info('[Epoch {} Loader {}/{}] NPML train loss: {}'.format(
                epoch, i, len(dataloaders), loss.detach().item()))

        torch.cuda.empty_cache()
        return

    def _evaluate_once(self, epoch):
        with torch.no_grad():
            util.to_eval(self.models)
            dataloader = dataloader_builder.build_test(
                self.data_config, self.tasks, self.logger)
            context_dataloader, target_dataloader =\
                dataloader['context'], dataloader['target']

            # Process the context set as a whole (It should all be loaded onto memory)
            context_inputs, context_labels = [], []
            for (context_input, context_label) in context_dataloader:
                context_input, context_label =\
                        context_input.to(self.device), context_label.to(self.device)

                context_inputs.append(self.models['x_encoder'](context_input))
                context_labels.append(context_label)
            context_inputs = torch.stack(context_inputs).permute(1, 0, 2)
            context_labels = torch.stack(context_labels).unsqueeze(0)

            # Now go through batches of test set and compute the losses
            loss = 0
            num_batches = 0
            for (target_input, target_label) in target_dataloader:
                target_input, target_label =\
                        target_input.to(self.device), target_label.to(self.device)

                target_input = self.models['x_encoder'](target_input).unsqueeze(0)
                target_label = target_label.unsqueeze(0).unsqueeze(2)

                output = self.models['np'](
                    context_inputs, context_labels, target_input, target_label)

                # Compute R2 score
                if 'r2' in self.evaluators:
                    self.evaluators['r2'].update(output, target_label)

                # Compute MAE score
                if 'mae' in self.evaluators:
                    self.evaluators['mae'].update(output, target_label)

                # Compute test loss
                loss += self.criterion(output, target_label)
                num_batches += 1

            self.logger.info(
                '[Epoch {}] - NPML test loss: {:4f}'.format(epoch, loss.detach().item()/num_batches))

            eval_metrics = {}
            for standard in self.evaluators:
                metric = self.evaluators[standard].compute()
                eval_metrics[standard] = metric
                self.logger.infov(
                    '[Epoch {}] - NPML {} {} score: {:4f}'.format(
                        epoch, loss.detach().item()/num_batches, standard, metric))
                self.evaluators[standard].reset()

            return eval_metrics

