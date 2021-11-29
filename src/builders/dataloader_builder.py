import random
from math import floor
from torch.utils.data import random_split, DataLoader
from src.core.datasets import custom_collate_fn


def build(tasks, logger):
    # Randomly sample a task
    task_name = random.choice(list(tasks.keys()))
    task = tasks[task_name]
    logger.info('Those chosen task is: ' + task_name)

    # Randomly choose a context set split size (at least 5% and at most 95%)
    context_split = random.uniform(0.05, 0.95)

    # Randomly split the task into context and target sets
    context_size = floor(context_split * len(task))
    target_size = len(task) - context_size
    context_dataset, target_dataset = random_split(task, [context_size, target_size])
    logger.info(
        'Using a context size of {} and a target size of {}.'.format(context_size, target_size))

    # Create data loaders
    context_dataloader = DataLoader(context_dataset, batch_size=context_size, collate_fn=custom_collate_fn)
    target_dataloader = DataLoader(target_dataset, batch_size=target_size, collate_fn=custom_collate_fn)

    dataloaders = {
        'context': context_dataloader,
        'target': target_dataloader
    }
    return dataloaders
