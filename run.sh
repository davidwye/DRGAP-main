#!/bin/bash

source activate python
nohup python -u run/winobias_llama_run.py &>> log_CFDetailPreambles/winobias_llama3_CFDetailPreambles_0_1_2.log &
wait $!

source activate python
nohup python -u run/winogender_llama_run.py &>> log_CFDetailPreambles/winogender_llama3_CFDetailPreambles_0_1_2.log &
wait $!

source activate python
nohup python -u run/gap_llama_run.py &>> log_CFDetailPreambles/gap_llama3_CFDetailPreambles_0_1_2.log &
wait $!

source activate python
nohup python -u run/BUG_llama_run.py &>> log_CFDetailPreambles/BUG_llama3_CFDetailPreambles_0_1_2.log &
wait $!