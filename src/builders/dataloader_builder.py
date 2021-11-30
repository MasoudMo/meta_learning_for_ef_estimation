import random
from math import floor, ceil
from torch.utils.data import random_split, DataLoader
from src.core.datasets import custom_collate_fn


def build_train(data_config, tasks, logger):
    # Randomly sample a task
    task_name = random.choice(list(tasks['train'].keys()))
    task = tasks[task_name]
    logger.info('Task {} is chosen'.format(task_name))

    # Randomly split the task into context and target sets
    max_samples = data_config['max_samples']
    context_ratio_range = data_config.get(['context_ratio_range'], [0.05, 0.95])
    context_split = random.uniform(*context_ratio_range)
    context_size = floor(context_split * len(task))
    target_size = len(task) - context_size

    if context_size > max_samples or target_size > max_samples:
        num_splits = ceil(max(context_size, target_size) / max_samples)

        context_split_size = ceil(context_size / num_splits)
        target_split_size = ceil(target_size / num_splits)

        splits = [context_split_size] * num_splits
        splits[-1] = splits[-1] - (context_size % num_splits)
        splits = splits + ([target_split_size] * num_splits)
        splits[-1] = splits[-1] - (target_size % num_splits)
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
    task_name = list(tasks['test'].keys())[0]
    task = tasks[task_name]
    logger.info('Test task {} is loaded'.format(task_name))

    # Randomly split the task into context and target sets
    context_size = floor(0.3 * len(task))
    target_size = len(task) - context_size

    context_dataset, target_dataset = random_split(task, [context_size, target_size])

    # Create data loaders
    context_dataloader = DataLoader(
        context_dataset, batch_size=1, collate_fn=custom_collate_fn)
    target_dataloader = DataLoader(
        target_dataset, batch_size=1, collate_fn=custom_collate_fn)

    dataloaders = {
        'context': context_dataloader,
        'target': target_dataloader
    }
    return dataloaders
