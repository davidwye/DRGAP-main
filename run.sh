#!/bin/bash

source activate python
nohup python -u run/winobias_llama_run.py &>> log/winobias_llama3.log &
wait $!

source activate python
nohup python -u run/winogender_llama_run.py &>> log/winogender_llama3.log &
wait $!

source activate python
nohup python -u run/gap_llama_run.py &>> log/gap_llama3.log &
wait $!

source activate python
nohup python -u run/BUG_llama_run.py &>> log/BUG_llama3.log &
wait $!
