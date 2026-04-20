# Ablations

`-` means the field is not explicitly set in the corresponding YAML file.

## Training Configuration

| Exp | File | framework | model_type | use_ot | use_cfg | use_ada_mask | ada_mask_max | const_mask_ratio | dit_decoder_depth | num_epochs | batch_size | learning_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp01 | `configs/ablations/exp01.yaml` | `flow_matching` | `dit_xl` | `true` | `true` | `true` | `0.75` | `-` | `-` | `400` | `16` | `0.0003` |
| exp02 | `configs/ablations/exp02.yaml` | `flow_matching` | `dit_l` | `true` | `true` | `true` | `0.75` | `-` | `-` | `200` | `24` | `0.0003` |
| exp03 | `configs/ablations/exp03.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `true` | `0.75` | `-` | `-` | `300` | `32` | `0.0003` |
| exp04 | `configs/ablations/exp04.yaml` | `flow_matching` | `dit_s` | `true` | `true` | `true` | `0.75` | `-` | `-` | `300` | `64` | `0.0004` |
| exp05 | `configs/ablations/exp05.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `-` | `-` | `0.5` | `-` | `300` | `32` | `0.0003` |
| exp06 | `configs/ablations/exp06.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `true` | `0.5` | `-` | `-` | `300` | `32` | `0.0003` |
| exp07 | `configs/ablations/exp07.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `true` | `0.9` | `-` | `-` | `300` | `32` | `0.0003` |
| exp08 | `configs/ablations/exp08.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `true` | `0.75` | `-` | `0` | `300` | `32` | `0.0003` |
| exp09 | `configs/ablations/exp09.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `false` | `-` | `-` | `-` | `300` | `32` | `0.0003` |
| exp10 | `configs/ablations/exp10.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `true` | `0.75` | `-` | `-` | `800` | `32` | `0.0003` |
| exp11 | `configs/ablations/exp11.yaml` | `flow_matching` | `dit_b` | `true` | `true` | `true` | `0.75` | `-` | `-` | `800` | `32` | `0.0003` |
| exp12 | `configs/ablations/exp12.yaml` | `flow_matching` | `dit_l` | `false` | `true` | `false` | `-` | `-` | `-` | `200` | `16` | `0.0003` |
| exp13 | `configs/ablations/exp13.yaml` | `flow_matching` | `dit_l` | `true` | `true` | `true` | `0.75` | `-` | `-` | `300` | `16` | `0.0003` |
| exp14 | `configs/ablations/exp14.yaml` | `flow_matching` | `dit_l` | `true` | `true` | `true` | `0.75` | `-` | `0` | `300` | `16` | `0.0003` |
| exp15 | `configs/ablations/exp15.yaml` | `flow_matching` | `unet` | `true` | `true` | `-` | `-` | `-` | `-` | `50` | `32` | `0.0001` |
| exp16 | `configs/ablations/exp16.yaml` | `flow_matching` | `unet` | `false` | `true` | `-` | `-` | `-` | `-` | `50` | `32` | `0.0001` |
| exp17 | `configs/ablations/exp17.yaml` | `flow_matching` | `dit_xl` | `false` | `true` | `false` | `-` | `-` | `-` | `100` | `8` | `0.0003` |
| exp18 | `configs/ablations/exp18.yaml` | `flow_matching` | `dit_xl` | `true` | `true` | `true` | `0.75` | `-` | `-` | `200` | `8` | `0.0003` |
| exp19 | `configs/ablations/exp19.yaml` | `ddpm` | `unet` | `-` | `true` | `-` | `-` | `-` | `-` | `50` | `32` | `0.0001` |
| exp20 | `configs/ablations/exp20.yaml` | `ddpm` | `unet` | `-` | `true` | `-` | `-` | `-` | `-` | `200` | `32` | `0.0001` |

## Sampling and Resource Configuration

| Exp | slurm.node | slurm.gpus | slurm.time | flow_solver | num_inference_steps_flow |
| --- | --- | --- | --- | --- | --- |
| exp01 | `h100-80` | `8` | `12:00:00` | `-` | `-` |
| exp02 | `h100-80` | `8` | `12:00:00` | `-` | `-` |
| exp03 | `h100-80` | `8` | `12:00:00` | `-` | `-` |
| exp04 | `h100-80` | `8` | `12:00:00` | `-` | `-` |
| exp05 | `h100-80` | `8` | `03:00:00` | `-` | `-` |
| exp06 | `h100-80` | `8` | `02:00:00` | `-` | `-` |
| exp07 | `h100-80` | `8` | `01:00:00` | `-` | `-` |
| exp08 | `h100-80` | `8` | `01:00:00` | `-` | `-` |
| exp09 | `h100-80` | `8` | `03:00:00` | `-` | `-` |
| exp10 | `h100-80` | `8` | `03:00:00` | `heun` | `10` |
| exp11 | `h100-80` | `8` | `03:00:00` | `heun` | `4` |
| exp12 | `h100-80` | `8` | `05:00:00` | `euler` | `10` |
| exp13 | `h100-80` | `8` | `03:00:00` | `heun` | `10` |
| exp14 | `h100-80` | `8` | `03:00:00` | `heun` | `10` |
| exp15 | `h100-80` | `3` | `04:00:00` | `-` | `-` |
| exp16 | `h100-80` | `8` | `01:00:00` | `-` | `-` |
| exp17 | `h100-80` | `8` | `06:00:00` | `heun` | `10` |
| exp18 | `h100-80` | `8` | `05:00:00` | `heun` | `10` |
| exp19 | `h100-80` | `8` | `01:00:00` | `-` | `-` |
| exp20 | `h100-80` | `8` | `01:00:00` | `-` | `-` |
