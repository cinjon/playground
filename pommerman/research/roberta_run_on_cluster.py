"""Run on the cluster.

NOTE: See local_config.template.py for a local_config template.
"""
import os
import sys
import itertools
import local_config

directory = local_config.cluster_directory
email = local_config.email

slurm_logs = os.path.join(directory, "slurm_logs")
slurm_scripts = os.path.join(directory, "slurm_scripts")

if not os.path.exists(slurm_logs):
    os.makedirs(slurm_logs)
if not os.path.exists(slurm_scripts):
    os.makedirs(slurm_scripts)

abbr = {
    'lr': 'lr',
    'how-train': 'ht-',
    'num-steps': 'ns',
    'distill-epochs': 'dstlepi',
    'num-battles-eval': 'nbe',
    'gamma': 'gma',
    'set-distill-kl': 'sdkl',
    'num-processes': 'np',
    'config': 'cfg-',
    'model-str': 'm-',
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
    'stop-grads-value': 'sgv',
    'add-nonlin-valhead': 'anv',
    "batch-size": 'bs',
    "ppo-epoch": 'ep',
}

def train_ppo_job(flags, jobname=None):
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
        f.write("#SBATCH --qos=batch" + "\n")
        f.write("#SBATCH --mail-type=END,FAIL" + "\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("module purge" + "\n")
        local_config.write_extra_sbatch_commands(f)
        f.write(jobcommand + "\n")

    s = "sbatch --qos batch --gres=gpu:1 --nodes=1 "
    s += "--cpus-per-task=%s " % num_processes
    s += "--mem=256000 --time=48:00:00 %s &" % os.path.join(
        slurm_scripts, jobnameattrs + ".slurm")
    os.system(s)


def train_dagger_job(flags, jobname=None):
    num_processes = flags["num-processes"]
    jobname = 'pmandag'
    jobnameattrs = '%s.%s' % (
        jobname, '.'.join(['%s%s' % (abbr[k], str(flags[k])) for k in sorted(flags.keys()) if k in abbr])
    )
    jobcommand = "python train_dagger.py "
    args = ["--%s %s" % (flag, str(flags[flag])) for flag in sorted(flags.keys())]
    jobcommand += " ".join(args)
    print(jobcommand)

    slurmfile = os.path.join(slurm_scripts, jobnameattrs + '.slurm')
    with open(slurmfile, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name" + "=" + jobname + "\n")
        f.write("#SBATCH --output=%s\n" % os.path.join(slurm_logs, jobnameattrs + ".out"))
        f.write("#SBATCH --error=%s\n" % os.path.join(slurm_logs, jobnameattrs + ".err"))
        f.write("#SBATCH --qos=batch" + "\n")
        f.write("#SBATCH --mail-type=END,FAIL" + "\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("module purge" + "\n")
        local_config.write_extra_sbatch_commands(f)
        f.write(jobcommand + "\n")

    s = "sbatch --qos batch --gres=gpu:1 --nodes=1 "
    s += "--cpus-per-task=%s " % num_processes
    s += "--mem=64000 --time=48:00:00 %s &" % os.path.join(
        slurm_scripts, jobnameattrs + ".slurm")
    os.system(s)


################ JOBS: top to bottom = most recent to oldest  ####################

### May 15th ###

### use reverse state as curriculum

### Train PPO + LSTM
train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "1.0", "lr": 0.0001,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "0.995", "lr": 0.0001,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "0.99", "lr": 0.0001,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "0.95", "lr": 0.0001,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "1.0", "lr": 0.0001,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "0.995", "lr": 0.00005,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "0.99", "lr": 0.00005,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "lstm", "how-train": "simple", "log-interval": 1,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFA8x8-v0", "gamma": "0.95", "lr": 0.00005,
     "model-str": "PommeCNNPolicySmall",
     "board_size": 8, "batch-size": 5120, "save-interval": 100,
     "recurrent-policy": "", "how-train": "simple", "eval-mode": "ffa", "num-stack": 1,
    }, "lstm"
)

# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".90", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11, "batch-size": 5120, "save-interval": 100,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".99", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11, "batch-size": 5120, "save-interval": 100,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".95", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11,  "batch-size": 5120, "use-gae": "", "save-interval": 100,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )
#
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".95", "lr": 0.00007,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11, "batch-size": 5120, "save-interval": 100,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".95", "lr": 0.00005,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11, "batch-size": 5120, "save-interval": 100,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".95", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11, "batch-size": 5120, "save-interval": 100,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".95", "lr": 0.0005,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11, "batch-size": 5120, "save-interval": 100,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )

# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "overfit", "how-train": "simple", "log-interval": 1,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".95", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall",
#      "board_size": 11, "batch-size": 8124,
#      "state-directory": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/pomme_games/ffaeasyv0-seed1",
#      "state-directory-distribution": "backloaded",
#     }, "overfit"
# )


# ### May 9 ###
#
# ### easier configs
#
# ### Train PPO + Distilling for FFA -- high kl_facto 10x distill_factor with distill_epoch 10k
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo-dstl", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v3", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#      "board_size": 11,
#     }, "easy-ppo-dstl"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo-dstl", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#      "board_size": 11,
#     }, "easy-ppo-dstl"
# )
#
# ### Train PPO + Distilling for FFA -- high kl_facto fixed to 10
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo-dstl10", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v3", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#      "set-distill-kl": 10,
#      "board_size": 11,
#     }, "easy-ppo-dstl10"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo-dstl10", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#      "set-distill-kl": 10,
#      "board_size": 11,
#     }, "easy-ppo-dstl10"
# )
#
#
# ### Jobs with PPO but no distillation: FFA, both v0 and v3; only Small net.
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v3", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "board_size": 11,
#     }, "easy-ppo"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAEasy-v0", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "board_size": 11,
#     }, "easy-ppo"
# )
#
# # PPO Team Small
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeTeamEasy-v3", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "board_size": 11,
#     }, "easy-ppo"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeTeamEasy-v0", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "board_size": 11,
#     }, "easy-ppo"
# )
#
# # PPO Team Smaller
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeTeamEasy-v3", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "board_size": 11,
#     }, "easy-ppo"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "easy-ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeTeamEasy-v0", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "board_size": 11,
#     }, "easy-ppo"
# )



### trying a different HP set - similar to Cinjon's one and my initial ones
# smaller minibatch-size (1000/500 -> 275), larger num-episodes-dagger (20 -> 30),
# smaller lr (0.001 -> 0.0007) lr 0.001

### These are trying to dagger train an agent in FFA w/ value loss game. After Dagger was put on multiprocesses.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "newest-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 250, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "",
#      }, "newest-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "newest-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 250, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "newest-dag-val"
# )
#
#
#
# ### These are trying to dagger train an agent in FFA with value loss and stopgrads
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "newest-dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 250, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "newest-dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "newest-dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 250, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "newest-dag-val-stop"
# )
#
#
# ### These are trying to dagger train an agent in FFA with value loss and stopgrads
# # They also have extra hidden and nonlinear layer in value head only.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallerNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "newest-dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 250, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "newest-dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallerNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "newest-dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "newest-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 30, "log-interval": 50,
#      "minibatch-size": 250, "save-interval": 50, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "newest-dag-val-stop-nonlin"
# )



### May 8 ### trying a different HP set - similar to Cinjon's one and my initial ones
# smaller minibatch-size (1000/500 -> 275), larger num-episodes-dagger (2 -> 20),
# larger lr (0.001 -> 0.005)
# lr 0.001 better than 0.005 and seemed to have helped to increase num-episodes-dagger
# but not that much

### These are trying to dagger train an agent in FFA w/ value loss game. After Dagger was put on multiprocesses.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "new-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "",
#      }, "new-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "new-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "new-dag-val"
# )
#
# ### These are trying to dagger train an agent in FFA w/ value loss game. After Dagger was put on multiprocesses.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "new-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "",
#      }, "new-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "new-dag-val"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "new-dag-val"
# )
#
#
# ### These are trying to dagger train an agent in FFA with value loss and stopgrads
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "new-dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "new-dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "new-dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "new-dag-val-stop"
# )
#
#
# ### These are trying to dagger train an agent in FFA with value loss and stopgrads
# # They also have extra hidden and nonlinear layer in value head only.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallerNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "new-dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "new-dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallerNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "new-dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "new-dag-val-stop-nonlin"
# )

# TODO: DID NOT RUN THESE ONES YET
### These are trying to dagger train an agent in FFA w/out value loss game. After Dagger was put on multiprocesses.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-noval", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-noval", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-noval", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "new-dag-noval", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 125, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "dag-noval"
# )


### May 7 ### none of them works!!! still below 10% success rate!!

### These are trying to dagger train an agent in FFA w/out value loss game. After Dagger was put on multiprocesses.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "dag-noval"
# )
#
# ### These are trying to dagger train an agent in FFA w/ value loss game. After Dagger was put on multiprocesses.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "",
#      }, "dag-noval"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-noval", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "dag-noval"
# )
#
# ### These are trying to dagger train an agent in FFA with value loss and stopgrads
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "dag-val-stop"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      "use-value-loss": "", "stop-grads-value": "",
#      }, "dag-val-stop"
# )
#
#
# ### These are trying to dagger train an agent in FFA with value loss and stopgrads
# # They also have extra hidden and nonlinear layer in value head only.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallerNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallerNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "dag-val-stop-nonlin"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dag-val-stop-nonlin", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmallNonlinCritic",
#      "use-value-loss": "", "stop-grads-value": "", "add-nonlin-valhead": "",
#      }, "dag-val-stop-nonlin"
# )







### These are trying to dagger train an agent in FFA w/out value loss game. After Dagger was put on multiprocesses.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag-novalnet"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag-novalnet"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag-novalnet"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag-novalnet"
# )
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag-novalnet"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag-novalnet"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag-novalnet"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag-novalnet", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag-novalnet"
# )


### May 5 ###

### Train PPO + Distilling for FFA -- high kl_facto 10x distill_factor with distill_epoch 10k
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "kl10fix", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanKL10fix"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "kl10dstl", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanKL10dstl"
# )

# ### Train PPO + Distilling for FFA -- high kl_facto fixed to 10
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "kl10fix", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#      "set-distill-kl": 10,
#     }, "pmanKL10fix"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "kl10fix", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#      "set-distill-kl": 10,
#     }, "pmanKL10fix"
# )





### These are trying to dagger train an agent in FFA with value loss game. After Dagger was put on multiprocesses
###
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
#
#
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 2, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )


#### May 5th ####

### Jobs with PPO but no distillation: FFA, both v0 and v3; only Small net.
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#     }, "pmanPPO"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#     }, "pmanPPO"
# )
#
#
# ### Jobs with reinforce-only (no PPO): FFA, both distill from Simple and not distill, \
# ### both v0 and v3; only Small net.
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "pg-dstl", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "reinforce-only": "",
#     }, "pmanPGdstl"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "pg-dstl", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "reinforce-only": "",
#     }, "pmanPGdstl"
# )
#
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "pg", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "reinforce-only": "",
#     }, "pmanPGdstl"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "pg", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "reinforce-only": "",
#     }, "pmanPGdstl"
# )

### These are meant to see if we can distill the 1hot SImple Agent into the PPO while playing Team. These all have a bigger batch size.
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )


#
# ### This is just like above but without value loss
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100,
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
