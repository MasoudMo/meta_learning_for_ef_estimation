import random
from math import floor, ceil
from torch.utils.data import random_split, DataLoader
from src.core.datasets import custom_collate_fn
import numpy as np


def build_train(data_config, tasks, logger):
    # Randomly sample a task
    task_name = random.choice(list(tasks['train'].keys()))
    task = tasks['train'][task_name]
    logger.info('Task {} is chosen'.format(task_name))

    # Randomly split the task into context and target sets
    max_samples = data_config['max_samples']
    context_ratio_range = data_config.get('context_ratio_range', [0.05, 0.95])
    context_split = random.uniform(*context_ratio_range)
    context_size = floor(context_split * len(task))
    target_size = len(task) - context_size

    if context_size > max_samples or target_size > max_samples:
        num_splits = ceil(max(context_size, target_size) / max_samples)

        context_split_size = floor(context_size / num_splits)
        target_split_size = floor(target_size / num_splits)

        context_splits = np.array([context_split_size] * num_splits, dtype=np.int)
        offset = np.zeros(num_splits, dtype=np.int)
        offset[0:int(context_size - context_split_size * num_splits)] = 1
        context_splits = context_splits + offset
        context_splits = context_splits.tolist()

        target_splits = np.array([target_split_size] * num_splits, dtype=np.int)
        offset = np.zeros(num_splits, dtype=np.int)
        offset[0:int(target_size - target_split_size * num_splits)] = 1
        target_splits = target_splits + offset
        target_splits = target_splits.tolist()

        splits = context_splits + target_splits
    else:
        num_splits = 1
        splits = [context_size, target_size]

    datasets = random_split(task, splits)
    context_datasets = datasets[:num_splits]
    target_datasets = datasets[num_splits:]

    logger.info(
        'Using a context size of {} and a target size of {}.'.format(context_size, target_size))

    # Create data loaders
    dataloaders = []
    for context_dataset, target_dataset in zip(context_datasets, target_datasets):
        context_size, target_size = len(context_dataset), len(target_dataset)
        context_dataloader = DataLoader(context_dataset, batch_size=context_size, collate_fn=custom_collate_fn)
        target_dataloader = DataLoader(target_dataset, batch_size=target_size, collate_fn=custom_collate_fn)
        dataloaders.append(
            {'context': context_dataloader, 'target': target_dataloader})

    return dataloaders

def build_test(data_config, tasks, logger):
    max_samples = data_config['max_samples']
    task = tasks['test']
    logger.info('Test task is loaded')

    # Randomly split the task into context and target sets
    context_size = floor(0.001 * len(task))
    target_size = len(task) - context_size

    context_dataset, target_dataset = random_split(task, [context_size, target_size])

    # Create data loaders
    context_dataloader = DataLoader(
        context_dataset, batch_size=1, collate_fn=custom_collate_fn)
    target_dataloader = DataLoader(
        target_dataset, batch_size=max_samples, collate_fn=custom_collate_fn)

    dataloaders = {
        'context': context_dataloader,
        'target': target_dataloader
    }
    return dataloaders
