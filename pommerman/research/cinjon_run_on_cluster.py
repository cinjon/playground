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
}

def train_ppo_job(flags):
    num_processes = flags["num-processes"]
    jobname = 'pman'
    jobnameattrs = '%s.%s' % (
        jobname, '.'.join(['%s%s' % (abbr[k], str(flags[k])) for k in sorted(flags.keys()) if k in abbr])
    )
    jobcommand = "CUDA_VISIBLE_DEVICES=0 python train_ppo.py "
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


def train_dagger_job(flags):
    num_processes = flags["num-processes"]
    jobname = 'pmandag'
    jobnameattrs = '%s.%s' % (
        jobname, '.'.join(['%s%s' % (abbr[k], str(flags[k])) for k in sorted(flags.keys()) if k in abbr])
    )
    jobcommand = "CUDA_VISIBLE_DEVICES=0 python train_dagger.py "
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


### These were all attempts to do KL distilling into the PPO Policy. None of them worked well.
# train_ppo_job({
#     "num-processes": 8, 
#     "run-name": "fresh",
#     "how-train": "homogenous",
#     "num-steps": 100,
#     "log-interval": 150,
#     "lr": 0.0003,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 2500,
#     "distill-target": "dagger::/home/resnick/Code/selfplayground/thisone-dagger-save-eval-env-done-rand-seed.ht-dagger.cfg-PommeFFA-v3.m-PommeCNNPolicySmall.nc-256.lr-0.005-.mb-5000.ns-5000.num-0.epoch-580.steps-34860000.seed-0.pt",
#     "restart-counts": "",
#     "config": "PommeTeam-v0",
#     "eval-mode": "homogenous",
#     "num-battles-eval": 50,
#     "gamma": ".995"
# })

# train_ppo_job({
#     "num-processes": 8, 
#     "run-name": "fresh",
#     "how-train": "homogenous",
#     "num-steps": 100,
#     "log-interval": 150,
#     "lr": 0.0001,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 2500,
#     "distill-target": "dagger::/home/resnick/Code/selfplayground/thisone-dagger-save-eval-env-done-rand-seed.ht-dagger.cfg-PommeFFA-v3.m-PommeCNNPolicySmall.nc-256.lr-0.005-.mb-5000.ns-5000.num-0.epoch-580.steps-34860000.seed-0.pt",
#     "restart-counts": "",
#     "config": "PommeTeam-v0",
#     "eval-mode": "homogenous",
#     "num-battles-eval": 50,
#     "gamma": ".995"
# })

# train_ppo_job({
#     "num-processes": 8, 
#     "run-name": "fresh",
#     "how-train": "homogenous",
#     "num-steps": 100,
#     "log-interval": 150,
#     "lr": 0.0003,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 2000,
#     "distill-target": "dagger::/home/resnick/Code/selfplayground/thisone-dagger-save-eval-env-done-rand-seed.ht-dagger.cfg-PommeFFA-v3.m-PommeCNNPolicySmall.nc-256.lr-0.005-.mb-5000.ns-5000.num-0.epoch-580.steps-34860000.seed-0.pt",
#     "restart-counts": "",
#     "config": "PommeTeam-v0",
#     "eval-mode": "homogenous",
#     "num-battles-eval": 50,
#     "gamma": ".995"
# })

# train_ppo_job({
#     "num-processes": 8, 
#     "run-name": "fresh",
#     "how-train": "homogenous",
#     "num-steps": 100,
#     "log-interval": 150,
#     "lr": 0.0001,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 2000,
#     "distill-target": "dagger::/home/resnick/Code/selfplayground/thisone-dagger-save-eval-env-done-rand-seed.ht-dagger.cfg-PommeFFA-v3.m-PommeCNNPolicySmall.nc-256.lr-0.005-.mb-5000.ns-5000.num-0.epoch-580.steps-34860000.seed-0.pt",
#     "restart-counts": "",
#     "config": "PommeTeam-v0",
#     "eval-mode": "homogenous",
#     "num-battles-eval": 50,
#     "gamma": ".995"
# })

# train_ppo_job({
#     "num-processes": 8, 
#     "run-name": "fresh",
#     "how-train": "homogenous",
#     "num-steps": 100,
#     "log-interval": 150,
#     "lr": 0.0003,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "set-distill-kl": 1.0,
#     "distill-target": "dagger::/home/resnick/Code/selfplayground/thisone-dagger-save-eval-env-done-rand-seed.ht-dagger.cfg-PommeFFA-v3.m-PommeCNNPolicySmall.nc-256.lr-0.005-.mb-5000.ns-5000.num-0.epoch-580.steps-34860000.seed-0.pt",
#     "restart-counts": "",
#     "config": "PommeTeam-v0",
#     "eval-mode": "homogenous",
#     "num-battles-eval": 50,
#     "gamma": ".995"
# })

# train_ppo_job({
#     "num-processes": 8, 
#     "run-name": "fresh",
#     "how-train": "homogenous",
#     "num-steps": 100,
#     "log-interval": 150,
#     "lr": 0.0001,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "set-distill-kl": 1.0,
#     "distill-target": "dagger::/home/resnick/Code/selfplayground/thisone-dagger-save-eval-env-done-rand-seed.ht-dagger.cfg-PommeFFA-v3.m-PommeCNNPolicySmall.nc-256.lr-0.005-.mb-5000.ns-5000.num-0.epoch-580.steps-34860000.seed-0.pt",
#     "restart-counts": "",
#     "config": "PommeTeam-v0",
#     "eval-mode": "homogenous",
#     "num-battles-eval": 50,
#     "gamma": ".995"
# })

### These are trying to have PPO learn to play with a simple agent against two other simple agents.
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma995", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma995", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma995", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma995", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".99", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0001,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )
# train_ppo_job(
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller",
#     }
# )


### These are trying to dagger train a PPO agent in a TeamRandom game.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma99", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 25,
#      "minibatch-size": 2000, "save-interval": 25, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma995", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 25,
#      "minibatch-size": 2000, "save-interval": 25, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma99", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 25,
#      "minibatch-size": 2000, "save-interval": 25, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma995", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 25,
#      "minibatch-size": 2000, "save-interval": 25, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma99", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 10,
#      "minibatch-size": 2000, "save-interval": 10, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma995", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 10,
#      "minibatch-size": 2000, "save-interval": 10, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma99", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 10,
#      "minibatch-size": 2000, "save-interval": 10, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "dagg-gma995", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 10,
#      "minibatch-size": 2000, "save-interval": 10, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }
# )
