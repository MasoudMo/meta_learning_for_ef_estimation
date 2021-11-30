import os
from copy import deepcopy
from src.core import datasets

DATASETS = {
    'camus': datasets.CamusEfDataset,
    'echo': datasets.EchoNetEfDataset,
}

def build(data_config, logger):
    root = data_config['root']
    image_shape = 128 # default image_shape

    tasks = {'train': {}, 'test': {}}
    task_tmp_names = [key for key in data_config.keys() if 'task' in key]
    for task_tmp_name in task_tmp_names:
        task_config = deepcopy(data_config[task_tmp_name])
        data_name = task_config.pop('name')
        task_config['dataset_path'] = os.path.join(root, data_name)

        task_name = '_'.join([data_name, task_config['task'], task_config['view']])
        tasks['train'][task_name] = DATASETS[data_name](**task_config)
        image_shape = task_config['image_shape'] # store it for the test set

    tasks['test'] = DATASETS['camus'](image_shape=image_shape, task='all_ef', view='all_views')
    logger.infov(
        'Tasks are built - {}'.format(tasks))
    return tasks

