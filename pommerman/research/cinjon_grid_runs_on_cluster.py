"""Run on the cluster

NOTE: See local_config.template.py for a local_config template.
"""
import os
import sys
import itertools
import local_config

directory = os.path.join(local_config.cluster_directory, 'grid')
email = local_config.email

slurm_logs = os.path.join(directory, "slurm_logs")
slurm_scripts = os.path.join(directory, "slurm_scripts")

if not os.path.exists(slurm_logs):
    os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
    os.makedirs(slurm_scripts)

abbr = {
    'lr': 'lr',
    'board-size': 'bs',
    'how-train': 'ht-',
    'num-steps': 'ns',
    'distill-epochs': 'dstlepi',
    'num-battles-eval': 'nbe',
    'gamma': 'gma',
    'set-distill-kl': 'sdkl',
    'num-channels': 'nc',
    'num-processes': 'np',
    'config': 'cfg-',
    'model-str': 'm-',
    'num-mini-batch': 'nmbtch',
    'minibatch-size': 'mbs',
    'log-interval': 'log',
    'save-interval': 'sav',
    'expert-prob': 'exprob',
    'num-steps-eval': 'nse',
    'use-value-loss': 'uvl',
    'num-episodes-dagger': 'ned',
    'num-mini-batch': 'nmb',
    'use-lr-scheduler': 'ulrs',
    'half-lr-epochs': 'hlre',
    'use-gae': 'gae',
    'init-kl-factor': 'ikl',
    'state-directory-distribution': 'sdd',
    'anneal-bomb-penalty-epochs': 'abpe',
    'begin-selfbombing-epoch': 'bsbe',
    'item-reward': 'itr',
    'use-second-place': 'usp',    
}


def train_ppo_job(flags, jobname=None, is_fb=False, partition="uninterrupted"):
    num_processes = flags["num-processes"]
    jobname = jobname or 'pman'
    jobnameattrs = '%s.%s' % (
        jobname, '.'.join(['%s%s' % (abbr[k], str(flags[k])) for k in sorted(flags.keys()) if k in abbr])
    )
    jobcommand = "python train_ppo.py "
    args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
    jobcommand += " ".join(args)
    print(jobcommand)

    slurmfile = os.path.join(slurm_scripts, jobnameattrs + '.slurm')
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name" + "=" + jobname + "\n")
        f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobnameattrs + ".out"))
        f.write("#SBATCH --error=%s\n" % os.path.join(slurm_logs, jobnameattrs + ".err"))
        if is_fb:
            f.write("#SBATCH --partition=%s\n" % partition)
        else:
            f.write("#SBATCH --qos=batch\n")
        f.write("#SBATCH --mail-type=END,FAIL\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("module purge\n")
        local_config.write_extra_sbatch_commands(f)
        f.write(jobcommand + "\n")

    if is_fb:
        s = "sbatch --gres=gpu:1 --nodes=1 "
    else:
        s = "sbatch --qos batch --gres=gpu:1 --nodes=1 "        
    s += "--cpus-per-task=%s " % num_processes
    s += "--mem=64000 --time=24:00:00 %s &" % os.path.join(
        slurm_scripts, jobnameattrs + ".slurm")
    os.system(s)

### First djikstra runs.
job = {
    "how-train": "grid",  "log-interval": 5000, "save-interval": 100,
    "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
    "save-dir": os.path.join(directory, "models"), "num-channels": 32,
    "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
    "num-processes": 60, "gamma": 0.99, "batch-size": 25600, "num-mini-batch": 5,
    "state-directory":"/checkpoint/cinjon/selfplayground/gridastars110-s100/train",
}
counter = 0
for learning_rate in [1e-4, 3e-4, 1e-3]:
    for (name, distro) in [
            ("genesis", "genesis"),
            ("uBnGrA", "uniformBoundsGrA"),
            ("uBnGrB", "uniformBoundsGrB"),
    ]:
        for step_loss in [.03, .1]:
            for seed in [1]:
                j = {k:v for k,v in job.items()}
                j["run-name"] = "grid-%s-%d" % (name, counter)
                j["seed"] = seed
                j["state-directory-distribution"] = distro
                j["lr"] = learning_rate
                train_ppo_job(j, j["run-name"], is_fb=True, partition="learnfair")
                counter += 1
