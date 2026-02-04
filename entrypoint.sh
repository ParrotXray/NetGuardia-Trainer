#!/bin/bash
set -e
cd src/

CMD="./main.py"

if [ -n "$DATASET" ]; then
    CMD="$CMD --set $DATASET"
fi

[ "$ALL" = "true" ] && CMD="$CMD --all"
[ "$DATAPREPROCESS" = "true" ] && CMD="$CMD --datapreprocess"
[ "$DEEPAUTOENCODER" = "true" ] && CMD="$CMD --deepautoencoder"
[ "$MLP" = "true" ] && CMD="$CMD --mlp"
[ "$EXPORT" = "true" ] && CMD="$CMD --export"

exec $CMD "$@"