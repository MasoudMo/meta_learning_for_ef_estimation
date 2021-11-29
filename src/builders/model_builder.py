from copy import deepcopy
from functools import partial
import npf
from np.architectures import MLP, merge_flat_input
from src.core import models

ENCODERS = {
    'resnet2plus1d': models.Resnet2Plus1D
}


NP_MODLES = {
    'nlp': npf.LNP
}

def build(model_config, logger):
    encoder_config = model_config['encoder']
    encoder_name = encoder_config['name']
    r_dim = encoder_config['dim']
    encoder_num_layers = encoder_config['num_layers']

    decoder_config = deepcopy(model_config['decoder'])
    decoder_num_layers = decoder_config['num_layers']

    np_config = deepcopy(model_config['np'])
    np_model_name = np_config.pop('name', 'nlp')

    x_encoder = ENCODERS[encoder_name](output_dim=r_dim)
    xy_encoder = merge_flat_input(
        partial(MLP, n_hidden_layers=encoder_num_layers), is_sum_merge=True)
    decoder = merge_flat_input(
        partial(MLP, n_hidden_layers=decoder_num_layers), is_sum_merge=True)
    model = NP_MODLES[np_model_name](
        XYEncoder=xy_encoder, Decoder=decoder, **np_config)

    models = {
        'encoder': x_encoder,
        'np': model
    }

    logger.infov(
        'model is built - encoder: {}, decoder: MLP, NP: {}'.format(
            encoder_name, np_model_name))

    return models


