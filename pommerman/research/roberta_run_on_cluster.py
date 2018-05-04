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
    'use-gae': 'gae'
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
    s += "--mem=64000 --time=48:00:00 %s &" % os.path.join(
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

### Jobs with PPO but no distillation: FFA, both v0 and v3; only Small net.
train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
     "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
     "distill-expert": "SimpleAgent", "distill-epochs": 10000,
    }, "pmanPPO"
)
train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "ppo", "how-train": "simple", "num-steps": 200, "log-interval": 100,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
     "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
     "distill-expert": "SimpleAgent", "distill-epochs": 10000,
    }, "pmanPPO"
)


### Jobs with reinforce-only (no PPO): FFA, both distill from Simple and not distill, \
### both v0 and v3; only Small net.
train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "pg-dstl", "how-train": "simple", "num-steps": 200, "log-interval": 100,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
     "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
     "distill-expert": "SimpleAgent", "distill-epochs": 10000, "reinforce-only": "",
    }, "pmanPGdstl"
)
train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "pg-dstl", "how-train": "simple", "num-steps": 200, "log-interval": 100,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
     "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2,
     "distill-expert": "SimpleAgent", "distill-epochs": 10000, "reinforce-only": "",
    }, "pmanPGdstl"
)

train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "pg", "how-train": "simple", "num-steps": 200, "log-interval": 100,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFAShort-v3", "gamma": ".995", "lr": 0.0003,
     "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "reinforce-only": "",
    }, "pmanPGdstl"
)
train_ppo_job( # Batch size of 800
    {"num-processes": 8, "run-name": "pg", "how-train": "simple", "num-steps": 200, "log-interval": 100,
     "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
     "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
     "config": "PommeFFAShort-v0", "gamma": ".995", "lr": 0.0003,
     "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "reinforce-only": "",
    }, "pmanPGdstl"
)

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

### These are trying to dagger train an agent in FFA game. After Dagger was put on multiprocesses
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 1000, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 500, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/logs/",
#      "save-dir": "/misc/vlgscratch4/FergusGroup/raileanu/pommerman_spring18/models/",
#      "config": "PommeFFAShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#      }, "loldag"
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
