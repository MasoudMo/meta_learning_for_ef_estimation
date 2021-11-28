from data import CamusEfDataset, EchoNetEfDataset, LVBiplaneEFDataset
import torch
import logging
from model import CustomCNN3D
from npf.architectures import MLP, merge_flat_input
from npf import LNP
from functools import partial
from npf import NLLLossLNPF
from torch.utils.data import random_split, DataLoader
import random
from math import floor, ceil
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


def main():

    R_dim = 128
    epochs = 100
    max_samples = 15

    # Device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    # load task datasets
    tasks = dict({'echo_ap4_high_risk_ef': EchoNetEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/EchoNet-Dynamic/EchoNet-Dynamic',
                                                            device=device,
                                                            max_frames=30,
                                                            nth_frame=1,
                                                            task='high_risk_ef'),
                  'echo_ap4_medium_risk_ef': EchoNetEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/EchoNet-Dynamic/EchoNet-Dynamic',
                                                              device=device,
                                                              max_frames=30,
                                                              nth_frame=1,
                                                              task='medium_ef_risk'),
                  'echo_ap4_slight_risk_ef': EchoNetEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/EchoNet-Dynamic/EchoNet-Dynamic',
                                                              device=device,
                                                              max_frames=30,
                                                              nth_frame=1,
                                                              task='slight_ef_risk'),
                  'echo_ap4_normal_ef': EchoNetEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/EchoNet-Dynamic/EchoNet-Dynamic',
                                                         device=device,
                                                         max_frames=30,
                                                         nth_frame=1,
                                                         task='normal_ef'),
                  'lv_biplane_ap2_high_risk_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                    image_shape=128,
                                                                    device=device,
                                                                    raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                    task='high_risk_ef',
                                                                    view='ap2'),
                  'lv_biplane_ap4_high_risk_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                    image_shape=128,
                                                                    device=device,
                                                                    raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                    task='high_risk_ef',
                                                                    view='ap4'),
                  'lv_biplane_ap2_medium_risk_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                      image_shape=128,
                                                                      device=device,
                                                                      raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                      task='medium_ef_risk',
                                                                      view='ap2'),
                  'lv_biplane_ap4_medium_risk_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                      image_shape=128,
                                                                      device=device,
                                                                      raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                      task='medium_ef_risk',
                                                                      view='ap4'),
                  'lv_biplane_ap2_slight_risk_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                      image_shape=128,
                                                                      device=device,
                                                                      raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                      task='slight_ef_risk',
                                                                      view='ap2'),
                  'lv_biplane_ap4_slight_risk_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                      image_shape=128,
                                                                      device=device,
                                                                      raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                      task='slight_ef_risk',
                                                                      view='ap4'),
                  'lv_biplane_ap2_normal_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                 image_shape=128,
                                                                 device=device,
                                                                 raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                 task='normal_ef',
                                                                 view='ap2'),
                  'lv_biplane_ap4_normal_ef': LVBiplaneEFDataset(dataset_path='D:/Workspace/RCL/datasets/preprocessed',
                                                                 image_shape=128,
                                                                 device=device,
                                                                 raw_data_summary_csv='Biplane_LVEF_DataSummary.csv',
                                                                 task='normal_ef',
                                                                 view='ap4')})

    validation_task = CamusEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/camus',
                                     image_shape=128,
                                     device=device,
                                     task='all_ef',
                                     view='all_views')

    x_encoder = CustomCNN3D(input_dim=128,
                            n_conv_layers=3,
                            out_channels=[10, 60, 128],
                            kernel_sizes=3,
                            pool_sizes=2,
                            output_dim=128,
                            cnn_dropout_p=0,
                            fc_dropout_p=0)
    xy_encoder = merge_flat_input(partial(MLP, n_hidden_layers=2, hidden_size=R_dim*2), is_sum_merge=True)
    decoder = merge_flat_input(partial(MLP, n_hidden_layers=4, hidden_size=R_dim), is_sum_merge=True,)
    lnp_model = LNP(x_dim=128,
                    y_dim=1,
                    r_dim=R_dim,
                    encoded_path='latent',
                    is_q_zCct=False,
                    n_z_samples_train=1,
                    n_z_samples_test=32,
                    XYEncoder=xy_encoder,
                    Decoder=decoder)

    x_encoder = x_encoder.to(device)
    lnp_model = lnp_model.to(device)

    optimizer = torch.optim.Adam(list(lnp_model.parameters()) + list(x_encoder.parameters()), lr=0.00001)

    loss_func = NLLLossLNPF()

    for epoch in range(epochs):

        x_encoder.train()
        lnp_model.train()

        # Randomly sample a task
        task = random.choice(list(tasks.keys()))
        logger.info('The chosen task is: ' + task)
        task = tasks[task]

        # Randomly choose a context set split size (at least 5% and at most 95%)
        context_split = random.uniform(0.05, 0.95)

        # Randomly split the task into context and target sets
        context_size = floor(context_split * len(task))
        target_size = len(task) - context_size

        if context_size > max_samples or target_size > max_samples:
            num_splits = ceil(max(context_size, target_size) / max_samples)

            context_split_size = floor(context_size/num_splits)
            target_split_size = floor(target_size/num_splits)

            splits = [context_split_size] * num_splits
            splits[-1] = splits[-1] + (context_size % num_splits)
            splits = splits + ([target_split_size] * num_splits)
            splits[-1] = splits[-1] + (target_size % num_splits)

        else:
            num_splits = 1
            splits = [context_size, target_size]

        datasets = random_split(task, splits)
        context_datasets = datasets[:num_splits]
        target_datasets = datasets[num_splits:]
        del datasets
        logger.info('Using a context size of {} and a target size of {}.'.format(context_size, target_size))
        logger.info('Splitting data into {} split/splits'.format(num_splits))

        for i, (context_dataset, target_dataset) in enumerate(zip(context_datasets, target_datasets)):

            # Create data loaders
            context_dataloader = DataLoader(context_dataset, batch_size=1, collate_fn=custom_collate_fn)
            target_dataloader = DataLoader(target_dataset, batch_size=1, collate_fn=custom_collate_fn)

            optimizer.zero_grad()

            # Gather all context and target data
            x_context = list()
            y_context = list()
            x_target = list()
            y_target = list()
            for x, y in context_dataloader:
                x_context.append(x_encoder(x))
                y_context.append(y)
            x_context = torch.stack(x_context).permute(1, 0, 2)
            y_context = torch.stack(y_context).unsqueeze(0)

            for x, y in target_dataloader:
                x_target.append(x_encoder(x))
                y_target.append(y)
            x_target = torch.stack(x_target).permute(1, 0, 2)
            y_target = torch.stack(y_target).unsqueeze(0)

            # Run the LNP model
            output = lnp_model(x_context, y_context, x_target, y_target)

            # Compute the NPML objective
            loss = loss_func(output, y_target)

            # Back propagate
            loss.backward()

            # Do one optimization step
            optimizer.step()

            logger.info('Split {} NPML: {}'.format(i, loss.detach().item()))
            torch.cuda.empty_cache()

        x_encoder.eval()
        lnp_model.eval()

        with torch.no_grad():

            context_size = floor(0.3 * len(task))
            target_size = len(validation_task) - context_size

            context_dataset, target_dataset = random_split(validation_task, [context_size, target_size])

            # Create data loaders
            context_dataloader = DataLoader(context_dataset, batch_size=1, collate_fn=custom_collate_fn)
            target_dataloader = DataLoader(target_dataset, batch_size=1, collate_fn=custom_collate_fn)

            x_context = list()
            y_context = list()
            x_target = list()
            y_target = list()
            for x, y in context_dataloader:
                x_context.append(x_encoder(x))
                y_context.append(y)
            x_context = torch.stack(x_context).permute(1, 0, 2)
            y_context = torch.stack(y_context).unsqueeze(0)

            for x, y in target_dataloader:
                x_target.append(x_encoder(x))
                y_target.append(y)
            x_target = torch.stack(x_target).permute(1, 0, 2)
            y_target = torch.stack(y_target).unsqueeze(0)

            # Run the LNP model
            output = lnp_model(x_context, y_context, x_target, y_target)

            # Compute the NPML objective
            loss = loss_func(output, y_target)

            logger.info('Epoch {} Validation NPML: {}'.format(epoch, loss.detach().item()))


if __name__ == '__main__':
    main()
