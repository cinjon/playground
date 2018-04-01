import os
import sys
import itertools

dry_run = '--dry-run' in sys.argv

if not os.path.exists("slurm_logs"):
    os.makedirs("slurm_logs")

if not os.path.exists("slurm_scripts"):
    os.makedirs("slurm_scripts")

basename = "new"

grids = [
     {
        "seed" : [1,2,3,4],
     }
]

jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs += [{k: v for d in option_set for k, v in d.items()}
             for option_set in product_options]

if dry_run:
    print("NOT starting jobs:")
else:
    print("Starting jobs:")


merged_grid = {}
for grid in grids:
    for key in grid:
        merged_grid[key] = [] if key not in merged_grid else merged_grid[key]
        merged_grid[key] += grid[key]

varying_keys = {key for key in merged_grid if len(merged_grid[key]) > 1}

for job in jobs:
    jobname = basename
    flagstring = ""
    for flag in job:
        flagstring = flagstring + " --" + flag + " " + str(job[flag])
        if flag in varying_keys:
            jobname = jobname + "_" + flag + str(job[flag])

    jobcommand = "OMP_NUM_THREADS=1 python main.py --num-processes 16 --config ffa_v3 --how-train simple --save-dir /home/raileanu/pomme_logs/trained_models --log-interval 10 --save-interval 1000 --run-name new-stats" + flagstring

    print(jobcommand)

    with open('slurm_scripts/' + jobname + '.slurm', 'w') as slurmfile:
        slurmfile.write("#!/bin/bash\n")
        slurmfile.write("#SBATCH --job-name" + "=" + jobname + "\n")
        slurmfile.write("#SBATCH --output=slurm_logs/" + jobname + ".out\n")
        slurmfile.write("#SBATCH --error=slurm_logs/" + jobname + ".err\n")
        slurmfile.write("#SBATCH --qos=batch" + "\n")
        slurmfile.write("#SBATCH --mail-type=END,FAIL" + "\n")
        slurmfile.write("#SBATCH --mail-user=raileanu@cs.nyu.edu" + "\n")
        slurmfile.write("module purge" + "\n")
        slurmfile.write(jobcommand + "\n")


    if not dry_run:
        os.system((
            "sbatch --qos batch -N 1 -c 16 --mem=64000 "
            "--time=48:00:00 slurm_scripts/" + jobname + ".slurm &"))

        # if we want more than 1 node per job with num-processes 48
        # os.system((
        #     "sbatch --qos batch -N 3 -c 16 --mem=256000 "
        #     "--time=48:00:00 slurm_scripts/" + jobname + ".slurm &"))
