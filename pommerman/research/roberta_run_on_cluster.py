"""Run on the cluster

NOTE: See local_config.template.py for a local_config template.
"""
import os
import sys
import itertools
import local_config

directory = os.path.join(local_config.cluster_directory, 'florensa')
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


def train_ppo_job(flags, jobname=None, is_fb=False,
                  partition="uninterrupted", time=24):
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
    s += "--mem=64000 --time=%d:00:00 %s &" % (
        time, os.path.join(slurm_scripts, jobnameattrs + ".slurm"))
    os.system(s)

### Running everything seeds 1,2 for everything except 4,5 for reg-grUBnB for the ICLR paper.
### Reg is the dataset being the optimal one.
job = {
    "how-train": "grid",  "log-interval": 10, "save-interval": 1000,
    "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
    "save-dir": os.path.join(directory, "models"), "num-channels": 32,
    "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
    "num-processes": 8, "gamma": 0.99, "board-size": 24,
    "state-directory": os.path.join(directory, "astars110-s100"),
    "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000,
    "state-directory-distribution": "florensa", "genesis-epoch": 1750,
    "florensa-starts-dir": os.path.join(directory, "starts"),
    "step-loss": 0.03,
}
counter = 0
for state_directory in [
        "",
        "-5opt",
        "-10opt",
]:
    for (name, distro) in [
            ("florensa", "florensa"),
    ]:
        for seed in [4,5]:
            for learning_rate in [1e-3]:
                for step_loss in [.03]:
                    j = {k:v for k,v in job.items()}
                    j["run-name"] = "%s-%d" % (name, counter)
                    j["run-name"] = "iclr%d-florensa-%s" % (seed, j["run-name"])
                    j["run-name"] += state_directory
                    if state_directory == "":
                        j["state-directory"] = os.path.join(
                            directory, "astars110-s100", "train")
                        j["log-dir"] += "-100"
                    else:
                        j["state-directory"] += "%s/train" % state_directory
                    time = 48
                    j["seed"] = seed
                    j["step-loss"] = step_loss
                    j["state-directory-distribution"] = distro
                    j["lr"] = learning_rate
                    train_ppo_job(j, j["run-name"], is_fb=False, time=time)
                    counter += 1
