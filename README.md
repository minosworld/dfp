# Direct Future Prediction Agent for MINOS

This repository contains code based on the paper [Learning to Act by Predicting the Future](https://github.com/IntelVCL/DirectFuturePrediction) by Alexey Dosovitskiy and Vladlen Koltun, adapted for use with the MINOS simulator.

## Installing

Make sure you have the MINOS library installed on your system, and then follow the guide in [dfp](dfp) to install prerequisites for this code.

## Running

See the example experiment configurations in [experiments](experiments). You can train an agent from inside a specific configuration directory using:

```
python3 run_exp.py train --env_config objectgoal_suncg_sf --gpu
```

Then, testing can be done through:

```
python3 run_exp.py test --env_config objectgoal_suncg_sf --gpu --test_checkpoint checkpoints/<timestamp>
```

By replacing `test` with `show` you can visualize the test episodes.  Use `python3 run_exp.py --help` to get usage information for additional command line arguments.  Refer to the [MINOS](https://github.com/minosworld/minos) instructions for more details on parameters and environment configurations.
