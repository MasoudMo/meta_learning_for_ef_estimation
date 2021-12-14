import os
import shutil
import argparse
from src.engine import Engine
from src.utils.util import load_log

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="Path to a config")
    parser.add_argument('--save_dir', default='',
                        help='Path to dir to save checkpoints and logs')
    parser.add_argument('--eval_only', action='store_true',
                        help='run eval only using the given checkpoint path')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = load_log(args.save_dir)

    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    engine = Engine(
        config_path=args.config_path, logger=logger, save_dir=args.save_dir)

    if args.eval_only:
        engine.evaluate()
    else:
        engine.run()

