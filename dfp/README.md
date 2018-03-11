# Adaptation of code from the paper "Learning to Act by Predicting the Future" by Alexey Dosovitskiy and Vladlen Koltun for use in MINOS simulator.

If you use this code please cite the following paper in addition to the MINOS paper:

    @inproceedings{DK2017,
    author    = {Alexey Dosovitskiy and Vladlen Koltun},
    title     = {Learning to Act by Predicting the Future},
    booktitle = {International Conference on Learning Representations (ICLR)},
    year      = {2017}
    }

## Installing dependencies
Run `pip install -r requirements.txt`.

## Running
- If you have multiple gpus, make sure that only one is visible with

        export CUDA_VISIBLE_DEVICES=NGPU

    where NGPU is the number of GPU you want to use, or "" if you do not want to use a gpu

- For speeding things up you may want to prepend "taskset -c NCORE" before the command, where NCORE is the number of the core to be used, for example:

        taskset -c 1 python3 run_exp.py train

  When training with a GPU, one core seems to perform the best. Without a GPU, you may want 4 or 8 cores.
