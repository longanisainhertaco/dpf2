# Advanced Workflow Tutorial

Run DPF2 on a cluster, monitor progress, and collect results.

## 1. Submit to an HPC scheduler

The `dpf2 simulate` command can be launched from a batch script. With Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=dpf2
#SBATCH --time=00:30:00
#SBATCH --ntasks=4
#SBATCH --output=logs/%j.out

module load python
cd $SLURM_SUBMIT_DIR
srun dpf2 simulate -c config.json -o run --diagnostics
```

The job reads `config.json` (see [examples/config.json](../../examples/config.json) or the [configuration templates](../config_templates/)) and writes diagnostics to the `run` directory.

## 2. Stream live diagnostics

For quick feedback, enable the live plot option:

```bash
dpf2 simulate -c config.json -o run --live-plot
```

The terminal streams current and voltage traces during execution. Use `--verbose` for detailed logging.

## 3. Retrieve results

When the job completes, copy the run directory back and extract waveforms:

```bash
scp -r mycluster:~/runs/latest ./run
dpf2 diagnostics --history run/history.json --current --voltage > traces.json
```

You can also produce plots directly:

```bash
dpf2 plot-run --history run/history.json --output current.png
```

See the [HPC design notes](../HPC_DESIGN.md) for more on scaling and cluster integration.
