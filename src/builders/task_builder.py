import os
from copy import deepcopy
from src.core import datasets

DATASETS = {
    'camus': datasets.CamusEfDataset,
    'echonet': datasets.EchoNetEfDataset,
    'biplane_lvef': datasets.LVBiplaneEFDataset,
    'nat_lvef': datasets.NatEfDataset,
    'del_lvef': datasets.DelEfDataset,
    'poc_lvef': datasets.PocEfDataset
}

def build(data_config, logger, device):
    root = data_config['root']

    tasks = {'train': {}, 'test': {}}
    dataset_tmp_names = [key for key in data_config.keys() if 'dataset' in key]
    for dataset_tmp_name in dataset_tmp_names:
        task_config = deepcopy(data_config[dataset_tmp_name])
        data_name = task_config.pop('name')
        task_config['datasets_root_path'] = root

        for view in task_config['view']:
            for label_type in task_config['task']:
                task_name = '_'.join([data_name, label_type, view])
                if task_name not in data_config['tasks_to_exclude']:
                    tasks['train'][task_name] = DATASETS[data_name](**dict(task_config,
                                                                           view=view,
                                                                           task=label_type,
                                                                           device=device))

    # Load the validation dataset
    task_config = deepcopy(data_config['valset'])
    tasks['test'] = DATASETS[task_config['name']](image_shape=task_config['image_shape'],
                                                  num_frames=task_config['num_frames'],
                                                  device=device,
                                                  task='all_ef',
                                                  view='ap4',
                                                  datasets_root_path=root)
    logger.infov(
        'Tasks are built - {}'.format(tasks))
    return tasks

