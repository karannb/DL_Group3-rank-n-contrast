#!/bin/bash

# kills stray w&b processes (sometimes)

USERNAME=f20212582
PATTERN=wandb
pgrep -u $USERNAME -f "^$PATTERN" | while read PID; do
    echo "Killing process ID $PID"
    kill $PID
done