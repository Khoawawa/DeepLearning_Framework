#!/bin/bash
set -e

BRANCH=${1:-main}
CONFIG=${2:-configs/config.yaml}

if [ ! -d DeepLearning_Framework ]; then
    git clone -b "$BRANCH" https://github.com/Khoawawa/DeepLearning_Framework.git
    cd DeepLearning_Framework
else
    cd DeepLearning_Framework
    git fetch origin
    git checkout "$BRANCH"
    git reset --hard origin/"$BRANCH"
fi
if [requirements.txt -nt .last_install ]; then
    pip install -q -r requirements.txt
    touch .last_install
fi

python main.py --config "$CONFIG"
