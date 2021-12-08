#!/bin/sh

while [[ $# -gt 1 ]]
  do
    key="$1"

    case $key in
      -c|--config_path)
      CONFIG_PATH="$2"
      shift # past argument
      ;;
      -s|--save_dir)
      SAVE_DIR="$2"
      shift # past argument
      ;;
      -e|--eval_only)
      EVAL_ONLY="$2"
      *) # unknown option
      ;;
    esac
  shift # past argument or value
  done

python run.py \
    --config_path=${CONFIG_PATH} \
    --save_dir=${SAVE_DIR} \
    --eval_only=${EVAL_ONLY}
