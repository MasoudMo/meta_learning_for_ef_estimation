## Install

### Pip

```
# clone repo
pip install -r requirements.txt
```

## Configs
Configs are written in the form of yaml. Please refer to the configs/default.yml for the details about how to structure configs.

## Datasets
It assumes datasets are located in data root which can be modified in the data section of a config file. we recommend to create symbolic links for datasets in the data directory.

## Train
To train a model, refer to the command below. Note that save_dir is a directory where all the checkpoints and logs are saved.
```bash
bash scripts/run.sh --config_path configs/default.yml --save_dir path/to/save/dir
```

## Eval
To evaluate a model, refer to the command below. Note that save_dir should be syncronized with the save_dir used for training.
```bash
bash scripts/run.sh --config_path configs/default.yml --save_dir path/to/save/dir --eval_only true
```
