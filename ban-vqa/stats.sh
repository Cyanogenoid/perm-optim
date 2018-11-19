#!/bin/bash

set -x
python collect.py baseline-{1..10}.txt
python collect.py po-weight-max-objects-baseline-{1..10}.txt
python collect.py po-weight-max-objects-{1..10}.txt
