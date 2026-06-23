
## Checkpoints

| Model | File Size | Update Date  | Valid MAE on PCQM4Mv2 | Download Link                                            |
| ----- | --------- | ------------ | --------------------- | -------------------------------------------------------- |
| L12   | 189MB     | Oct 04, 2022 | 0.0785                | https://1drv.ms/u/s!AgZyC7AzHtDBdWUZttg6N2TsOxw?e=sUOhox |
| L12_old | 189MB   | Mar 31, 2023 | 0.0787                | https://1drv.ms/u/s!AgZyC7AzHtDBesDk9tZK1yvbtzE?e=5H91Zq |

```shell
# create paths to checkpoints for evaluation

# download the above model weights (L12.pt, L18.pt) to ./
mkdir -p logs/L12
mkdir -p logs/L18
mv L12.pt logs/L12/
mv L18.pt logs/L18/
```

## Downstream Task -- (QM9/Moleculenet/MoleculeACE)
Download the checkpoint: L12-old.pt
```shell
export ckpt_path='./L12-old.pt'                # path to checkpoints
bash finetune_xxx.sh
```
