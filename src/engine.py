import torch
import time
from src.utils import util
from src.builders import model_builder, task_builder, dataloader_builder,\
    optimizer_builder, criterion_builder, checkpointer_builder, evaluator_builder,\
    meter_builder, scheduler_builder
import wandb
import matplotlib.pyplot as plt
import numpy as np

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
        self._build(mode='val')
        metrics = self._evaluate_attention(0)
        print('hey')


class Engine(BaseEngine):

    def __init__(self, config_path, logger, save_dir):
        super(Engine, self).__init__(config_path, logger, save_dir)

    def _build(self, mode, init=False):
        # Build tasks
        self.tasks = task_builder.build(
            self.data_config, self.logger, self.device)

        # Build a model
        self.model = model_builder.build(
            self.model_config, self.logger)
        self.model = self.model.to(self.device)

        # Build an optimizer, scheduler and criterion
        self.optimizer = optimizer_builder.build(
            self.train_config['optimizer'], self.model, self.logger)
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
            self.save_dir, self.logger, self.model, self.optimizer,
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
            train_metrics = self._train_one_epoch(epoch)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr()[0]

            self.logger.infov(
                '[Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(epoch, lr, train_time, self.loss_meter.avg))

            # Evaluate every Nth epochs
            if epoch % 1 == 0:
                eval_metrics = self._evaluate_once(epoch)
                self.checkpointer.save(epoch, eval_metrics)
                self.logger.info(
                    '[Epoch {}] - Test {}: {:4f}'.format(
                        epoch, self.eval_standard, eval_metrics[self.eval_standard]))
                self.logger.info(
                    '[Epoch {}] - best Test {}: {:4f}'.format(
                        epoch, self.eval_standard, self.checkpointer.best_eval_metric))


                wandb.log({'val_mae_loss': eval_metrics['mae'],
                           'val_r2_score': eval_metrics['r2']}, step=epoch)


            wandb.log({'train_npml_loss': self.loss_meter.avg,
                       'train_mae_loss': train_metrics['mae'],
                       'train_r2_score': train_metrics['r2']}, step=epoch)

            self.scheduler.step()
            self.loss_meter.reset()

    def _train_one_epoch(self, epoch):
        util.to_train(self.model)
        self.criterion.train()
        dataloaders = dataloader_builder.build_train(
            self.data_config, self.tasks, self.logger)

        for i, dataloader in enumerate(dataloaders):
            context_dataloader, target_dataloader = dataloader['context'], dataloader['target']

            for j, ((context_input, context_label), (target_input, target_label)) in enumerate(
                zip(context_dataloader, target_dataloader)):
                context_input, context_label =\
                    context_input.to(self.device), context_label.to(self.device)
                target_input, target_label =\
                    target_input.to(self.device), target_label.to(self.device)

                output = self.model(context_input, context_label, target_input, target_label)

                # Compute the NPML objective
                loss = self.criterion(output, target_label, self.model.video_latent_var)
                self.loss_meter.update(loss.detach().item())

                # Back propagate
                loss.backward()

                # Do one optimization step
                self.optimizer.step()

                self.logger.info('[Epoch {} Loader {}/{}] NPML train loss: {}'.format(
                    epoch, i, len(dataloaders), loss.detach().item()))

                with torch.no_grad():
                    # Update R2 evaluator
                    if 'r2' in self.evaluators:
                        self.evaluators['r2'].update(output, target_label, self.model.video_latent_var)

                    # Update MAE evaluator
                    if 'mae' in self.evaluators:
                        self.evaluators['mae'].update(output, target_label, self.model.video_latent_var)

        torch.cuda.empty_cache()

        with torch.no_grad():
            train_metrics = {}
            for standard in self.evaluators:
                metric = self.evaluators[standard].compute()
                train_metrics[standard] = metric
                self.logger.infov(
                    '[Epoch {}] - Train - {} score: {:4f}'.format(
                        epoch, standard, metric))
                self.evaluators[standard].reset()

        return train_metrics

    def _evaluate_once(self, epoch):
        with torch.no_grad():
            util.to_eval(self.model)
            self.criterion.eval()
            dataloaders = dataloader_builder.build_test(
                self.data_config, self.tasks, self.logger)

            loss = []
            for i, dataloader in enumerate(dataloaders):

                context_dataloader, target_dataloader = dataloader['context'], dataloader['target']

                for (context_input, context_label) in context_dataloader:
                    context_input, context_label =\
                            context_input.to(self.device), context_label.to(self.device)

                # Now go through batches of test set and compute the losses
                for (target_input, target_label) in target_dataloader:
                    target_input, target_label =\
                            target_input.to(self.device), target_label.to(self.device)

                    output = self.model(context_input, context_label, target_input, target_label)

                    # Update R2 evaluator
                    if 'r2' in self.evaluators:
                        self.evaluators['r2'].update(output, target_label, self.model.video_latent_var)

                    # Update MAE evaluator
                    if 'mae' in self.evaluators:
                        self.evaluators['mae'].update(output, target_label,  self.model.video_latent_var)

                    # Compute test loss
                    loss.append(self.criterion(output, target_label, self.model.video_latent_var).detach().item())

            self.logger.info(
                '[Epoch {}] - NPML test loss: {:4f}'.format(epoch, sum(loss)/len(loss)))

            eval_metrics = {}
            for standard in self.evaluators:
                metric = self.evaluators[standard].compute()
                eval_metrics[standard] = metric
                self.logger.infov(
                    '[Epoch {}] - Test - {} score: {:4f}'.format(
                        epoch, standard, metric))
                self.evaluators[standard].reset()

            return eval_metrics


    def _evaluate_attention(self, epoch):
        with torch.no_grad():
            util.to_eval(self.model)
            self.criterion.eval()
            dataloaders = dataloader_builder.build_test(
                self.data_config, self.tasks, self.logger)

            loss = []
            for i, dataloader in enumerate(dataloaders):

                context_dataloader, target_dataloader = dataloader['context'], dataloader['target']

                for (context_input, context_label) in context_dataloader:
                    context_input, context_label =\
                            context_input.to(self.device), context_label.to(self.device)

                # Now go through batches of test set and compute the losses
                for idx, (target_input, target_label) in enumerate(target_dataloader):
                    target_input, target_label =\
                            target_input.to(self.device), target_label.to(self.device)

                    output = self.model(context_input, context_label, target_input, target_label)
                    variance = output[0].variance.detach().cpu().numpy()

                    attention_scores = self.model.attender.attn.detach().cpu().numpy()

                    # best_context_idx = np.argmax(attention_scores[0, 0, :])
                    # worst_context_idx = np.argmin(attention_scores[0, 0, :])
                    # if idx % 10 == 0:
                    #     for i in range(50):
                    #         plt.imsave('./qual/' + str(idx) + '_best_' + str(i) + '.png', context_input.detach().cpu().numpy()[best_context_idx, i, 0, :, :])
                    #         plt.imsave('./qual/' + str(idx) + '_worst_' + str(i) + '.png', context_input.detach().cpu().numpy()[worst_context_idx, i, 0, :, :])
                    #         plt.imsave('./qual/' + str(idx) + '_target_' + str(i) + '.png', context_input.detach().cpu().numpy()[0, i, 0, :, :])

            return None