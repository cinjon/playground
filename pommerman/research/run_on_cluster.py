"""Run on the cluster.

NOTE: See local_config.template.py for a local_config template.

Example: python run_on_cluster.py <run-name> <num-procs> <num-channels> \
  <learning-rate>
"""
import os
import sys
import itertools
import local_config

directory = local_config.cluster_directory
email = local_config.email

slurm_logs = os.path.join(directory, "slurm_logs")
slurm_scripts = os.path.join(directory, "slurm_scripts")

run_name = sys.argv[1]
num_procs = sys.argv[2]
num_channels = sys.argv[3]
learning_rate = sys.argv[4]
distill_epoch = sys.argv[5]
distill_target = sys.argv[6]
saved_paths = sys.argv[7]

dry_run = '--dry-run' in sys.argv
if dry_run:
    print("NOT starting jobs:")
else:
    print("Starting jobs:")
    if not os.path.exists(slurm_logs):
        os.makedirs(slurm_logs)
    if not os.path.exists(slurm_scripts):
        os.makedirs(slurm_scripts)


basename = "pman_%s_nc%s_np%s_lr%s_de%s" % (run_name, \
            num_channels, num_procs, learning_rate, distill_epoch)

grids = [
     {
        "seed" : [1],
     }
]

jobs = []
for grid in grids:
    individual_options = [[{key: value} for value in values]
                          for key, values in grid.items()]
    product_options = list(itertools.product(*individual_options))
    jobs.extend([{k: v for d in option_set for k, v in d.items()}
                 for option_set in product_options])

merged_grid = {}
for grid in grids:
    for key in grid:
        merged_grid[key] = [] if key not in merged_grid else merged_grid[key]
        merged_grid[key] += grid[key]

varying_keys = {key for key in merged_grid if len(merged_grid[key]) > 1}

args = [
    "--num-processes %s" % num_procs,
    # "--how-train simple",
    "--num-steps 1000 "
    # "--save-interval 1000 ",
    # "--log-interval 100 ",
    # "--config PommeFFA-v3 ",
    # "--num-channels %s" % num_channels,
    "--lr %s" % learning_rate,
    "--save-dir %s" % os.path.join(directory, "models"),
    "--log-dir %s" % os.path.join(directory, "logs"),
    "--distill-epoch %s" % distill_epoch,
    "--distill-target %s" % distill_target,
    "--saved-paths %s" % saved_paths
]

for job in jobs:
    jobname = basename

    flagstring = ""
    for flag in job:
        flagstring += " --%s %s" % (flag, str(job[flag]))
        if flag in varying_keys:
            jobname += "_%s%s" % (flag, str(job[flag]))

    job_args = args + ["--run-name %s" % jobname]
    jobcommand = "OMP_NUM_THREADS=1 python train_ppo.py %s%s" % (
        " ".join(job_args), flagstring)
    print(jobcommand)

    if not dry_run:
        slurmfile = os.path.join(slurm_scripts, jobname + '.slurm')
        with open(slurmfile, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name" + "=" + jobname + "\n")
            f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobname + ".out"))
            f.write("#SBATCH --error=%s\n" % os.path.join(slurm_logs, jobname + ".err"))
            f.write("#SBATCH --qos=batch" + "\n")
            f.write("#SBATCH --mail-type=END,FAIL" + "\n")
            f.write("#SBATCH --mail-user=%s\n" % email)
            f.write("module purge" + "\n")
            local_config.write_extra_sbatch_commands(f)
            f.write(jobcommand + "\n")

        s = "sbatch --qos batch --gres=gpu:1 --nodes=1 "
        s += "--cpus-per-task=%s " % num_procs
        s += "--mem=64000 --time=48:00:00 %s &" % os.path.join(
            slurm_scripts, jobname + ".slurm")
        os.system(s)
