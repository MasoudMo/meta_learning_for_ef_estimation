from src.core.meters import AverageEpochMeter

def build(model_config, logger):
    loss_meter = AverageEpochMeter('loss meter', logger, fmt=':f')

    logger.infov('Loss and PR meters are built.')
    return loss_meter
