from data import CamusEfDataset, EchoNetEfDataset
import torch
import logging
from model import Resnet2Plus1D
from npf.architectures import MLP, merge_flat_input
from npf import LNP
from functools import partial
from npf import NLLLossLNPF
from torch.utils.data import random_split, DataLoader
import random
from math import floor
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

    # Device to use
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    # load task datasets
    tasks = dict({'camus_ap2_poor_ef': CamusEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/camus',
                                                      image_shape=128,
                                                      device=device,
                                                      task='high_risk_ef',
                                                      view='ap2'),
                  'camus_ap4_poor_ef': CamusEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/camus',
                                                      image_shape=128,
                                                      device=device,
                                                      task='high_risk_ef',
                                                      view='ap4'),
                  'camus_ap2_medium_ef': CamusEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/camus',
                                                        image_shape=128,
                                                        device=device,
                                                        task='medium_ef_risk',
                                                        view='ap2'),
                  'camus_ap4_medium_ef': CamusEfDataset(dataset_path='D:/Workspace/RCL/datasets/raw/camus',
                                                        image_shape=128,
                                                        device=device,
                                                        task='medium_ef_risk',
                                                        view='ap4')})

    x_encoder = Resnet2Plus1D(output_dim=R_dim)
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

    optimizer = torch.optim.Adam(list(lnp_model.parameters()) + list(x_encoder.parameters()), lr=0.00001)

    loss_func = NLLLossLNPF()

    for epoch in range(epochs):
        # Randomly sample a task
        task = random.choice(list(tasks.keys()))
        logger.info('Those chosen task is: ' + task)
        task = tasks[task]

        # Randomly choose a context set split size (at least 5% and at most 95%)
        context_split = random.uniform(0.05, 0.95)

        # Randomly split the task into context and target sets
        context_size = floor(context_split * len(task))
        target_size = len(task) - context_size
        context_dataset, target_dataset = random_split(task,
                                                       [context_size,
                                                        target_size])
        logger.info('Using a context size of {} and a target size of {}.'.format(context_size, target_size))

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

        logger.info('NPML: {}'.format(loss.detach().item()))


if __name__ == '__main__':
    main()
