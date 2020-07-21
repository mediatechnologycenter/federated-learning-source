#!/bin/bash
arr=( "$@" )
echo "Killing all running federated learning nodes and servers..."
pkill -f C88A33B946
echo "All good. All gone!"