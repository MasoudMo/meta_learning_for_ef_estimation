from copy import deepcopy
from functools import partial
import npf
from npf.architectures import MLP, merge_flat_input
from src.core import models
import torch.nn as nn

ENCODERS = {
    'resnet2plus1d': models.Resnet2Plus1D,
    'customcnn3d': models.CustomCNN3D,
    'customcnn3dspatial': models.CustomCNN3DSpatial
}


NP_MODLES = {
    'lnp': npf.LNP,
    'attnlnp': npf.AttnLNP
}

def build(model_config, logger):
    # Build a x_encoder
    x_encoder_config = deepcopy(model_config['x_encoder'])
    x_encoder_name = x_encoder_config.pop('name')
    x_encoder = partial(ENCODERS[x_encoder_name], **x_encoder_config)

    # Build a xy_encoder
    xy_encoder_config = model_config['xy_encoder']
    xy_encoder_num_layers = xy_encoder_config['num_layers']
    xy_encoder = merge_flat_input(partial(
        MLP, n_hidden_layers=xy_encoder_num_layers, hidden_size=xy_encoder_config['hidden_dim']), is_sum_merge=True)

    # Build a decoder
    decoder_config = deepcopy(model_config['decoder'])
    decoder_num_layers = decoder_config['num_layers']
    decoder_hidden_dim = decoder_config['hidden_dim']

    # Build a NP model
    np_config = deepcopy(model_config['np'])
    np_model_name = np_config.pop('name', 'lnp')

    decoder = merge_flat_input(
        partial(MLP,
                n_hidden_layers=decoder_num_layers,
                hidden_size=decoder_hidden_dim,
                activation=nn.Sigmoid()), is_sum_merge=True)
    model = NP_MODLES[np_model_name](
        XYEncoder=xy_encoder, Decoder=decoder, XEncoder=x_encoder, **np_config)

    logger.infov(
        'model is built - x_encoder: {}, xy_encoder: MLP, decoder: MLP, NP: {}'.format(
            x_encoder_name, np_model_name))

    return model

