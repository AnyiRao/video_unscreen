#!/bin/sh

script=$1
src=$2
tgt=$3

echo running src video ${src} target video ${tgt}

export PYTHONPATH=./
python tools/replace/${script}.py --src ${src} #--tgt ${tgt}
echo finished src video ${src} target video ${tgt}