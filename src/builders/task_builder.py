from copy import deepcopy
from src.core import datasets

DATASETS = {
    'camus': datasets.CamusEfDataset,
    'echo': datasets.EchoNetEfDataset,
}

def build(data_config, logger):
    root = data_config['root']

    tasks = {}
    task_tmp_names = [key for key in data_config.keys() if 'task' in key]
    for task_tmp_name in task_tmp_names:
        task_config = deepcopy(data_config[task_tmp_name])
        data_name = task_config.pop('name')
        task_config['dataset_path'] = root

        task_name = '_'.join([data_name, task_config['task'], task_config['view']])
        tasks[task_name] = DATASETS[data_name](**task_config)

    logger.infov(
        'Tasks are built - {}'.format(tasks))
    return tasks



