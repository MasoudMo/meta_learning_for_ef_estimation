from src.core.evaluators import R2Evaluator, MaeEvaluator

EVAL_TYPES = {
    'r2': R2Evaluator,
    'mae': MaeEvaluator,
}

def build(eval_config, logger):
    standards = eval_config['standards']
    evaluators = {}
    for standard in standards:
        evaluator = EVAL_TYPES[standard](logger)
        evaluators[standard] = evaluator

    logger.infov('Evaluator is build.')
    return evaluators
