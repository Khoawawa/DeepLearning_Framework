#!/bin/bash
set -e

BRANCH=${1:-main}
PROFILE=${2:-dev}
CONFIG=${3:-configs/config.yaml}

if [ ! -d DeepLearning_Framework ]; then
    git clone -b "$BRANCH" https://github.com/Khoawawa/DeepLearning_Framework.git
    cd DeepLearning_Framework
else
    cd DeepLearning_Framework
    git fetch origin
    git checkout "$BRANCH"
    git reset --hard origin/"$BRANCH"
fi

pip install -q -r requirements.txt

PROFILE="$PROFILE" python main.py --config "$CONFIG"
