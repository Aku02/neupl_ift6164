# NeuPL / MeanFlow experiments

JAX code for NeuPL-style meta-training on **JaxMARL** multi-agent Brax tasks plus Cartesian toy games for ablations.

## Dependencies

- **Cartesian games only**: `torch`, `numpy`.
- **JaxMARL / MA-Brax (MuJoCo-based envs)**: install per [FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL) (JAX, Brax, and related stack).

## JaxMARL MA-Brax runs


| Script | Role |
|--------|------|
| [`neupl_old.py`](neupl_old.py) | NeuPL–JPSRO with a **Gaussian** policy head (PPO-style updates). |
| [`meanflow_old.py`](meanflow_old.py) | **MeanFlow**-style generator with NFT / PPO-ratio training. |

Example commands:

```bash
python neupl_old.py --env-name halfcheetah_2x3 --wandb --wandb-project neupl-jpsro
python neupl_old.py --env-name 'humanoid_9|8' --wandb --wandb-project neupl-jpsro

python meanflow_old.py --env-name halfcheetah_2x3 --wandb --wandb-project meanflow-jpsro
python meanflow_old.py --env-name 'humanoid_9|8' --wandb --wandb-project meanflow-jpsro
```

`--env-name` choices include **`halfcheetah_2x3`** (default), **`halfcheetah_6x1`**, **`walker2d_2x3`**, and **`humanoid_9|8`**.

Other stuff we tried with different BR algos in this directory (e.g. `neupl_jpsro_cheetah.py`, `meanflow_jpsro_cheetah.py`, `flowpl_jpsro_cheetah.py`, `fppo_jpsro_cheetah.py`) follow the same JaxMARL layout and flags pattern.

## Cartesian games (sweep)

For **synthetic Cartesian games** (Gaussian / discrete-mixed / diffusion / MeanFlow policies; PSRO–Nash vs self-play), run:

```bash
bash run_cartesian_sweep.bash
```


## Acknowledgements

MeanFlow-related code and experiments draw on:

- [Aku02/diffusion_playground](https://github.com/Aku02/diffusion_playground)
- [HiccupRL/MeanFlowQL](https://github.com/HiccupRL/MeanFlowQL/tree/main) — official code for *One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow* (AAAI 2026).
- [irom-princeton/dppo](https://github.com/irom-princeton/dppo) — Diffusion Policy Policy Optimization (DPPO), useful as a reference for diffusion / policy-optimisation patterns.

Use the JaxMARL / Brax / MABrax papers and repositories for environment attribution.
