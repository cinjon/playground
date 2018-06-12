"""Run on the cluster

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
    'board-size': 'bs',
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
    'init-kl-factor': 'ikl',
    'state-directory-distribution': 'sdd',
    'anneal-bomb-penalty-epochs': 'abpe',
    'begin-selfbombing-epoch': 'bsbe'
}

def train_ppo_job(flags, jobname=None, is_fb=False):
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
            f.write("#SBATCH --partition=learnfair\n")
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
    s += "--mem=64000 --time=72:00:00 %s &" % os.path.join(
        slurm_scripts, jobnameattrs + ".slurm")
    os.system(s)


def train_dagger_job(flags, jobname=None, is_fb=False):
    num_processes = flags["num-processes"]
    jobname = jobname or 'pmandag'
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
        if is_fb:
            f.write("#SBATCH --partition=learnfair\n")
        else:
            f.write("#SBATCH --qos=batch\n")
        f.write("#SBATCH --mail-type=END,FAIL" + "\n")
        f.write("#SBATCH --mail-user=%s\n" % email)
        f.write("module purge" + "\n")
        local_config.write_extra_sbatch_commands(f)
        f.write(jobcommand + "\n")

    if is_fb:
        s = "sbatch --gres=gpu:1 --nodes=1 "
    else:
        s = "sbatch --qos batch --gres=gpu:1 --nodes=1 "        
    s += "--cpus-per-task=%s " % num_processes
    s += "--mem=64000 --time=48:00:00 %s &" % os.path.join(
        slurm_scripts, jobnameattrs + ".slurm")
    os.system(s)


### These were all attempts to do KL distilling into the PPO Policy. None of them worked well. 
### ... But they kinda shouldnt because they are using the FFA one.
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

### These are trying to have PPO learn to play with a simple agent against two other simple agents. None of them worked well.
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


# ### These are trying to dagger train an agent in a TeamRandom game. 
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 50, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#  }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#  }, "loldag"
# )

# ### These are like the above but don't use the value loss.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "lolexuvl", "how-train": "dagger", "num-episodes-dagger": 25, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, 
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "lolexuvl"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "lolexuvl", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, 
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "lolexuvl"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "lolexuvl", "how-train": "dagger", "num-episodes-dagger": 25, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, 
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#  }, "lolexuvl"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "lolexuvl", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.005, "num-steps-eval": 100, 
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#  }, "lolexuvl"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "loldag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "loldag", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.001, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "loldag"
# )


### These are ppo jobs where we "distill" the 1hot Simple Agent into the PPO Agent.
# train_ppo_job( # This job will have batch size of 800
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "num-steps": 200,
#      "distill-expert": "SimpleAgent"
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 400
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent"
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 200
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 4,
#      "distill-expert": "SimpleAgent"
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 300
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 4, "num-steps": 150,
#      "distill-expert": "SimpleAgent"
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 150
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "num-steps": 150,
#      "distill-expert": "SimpleAgent"
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 400
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 4, "num-steps": 200,
#      "distill-expert": "SimpleAgent"
#     }, "pmanDSSimp"
# )


### These are the same as above, but with a longer distill-epochs
# train_ppo_job( # This job will have batch size of 800
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "num-steps": 200,
#      "distill-expert": "SimpleAgent", "distill-epochs": 5000,
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 400
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2,
#      "distill-expert": "SimpleAgent", "distill-epochs": 5000,
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 200
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 4,
#      "distill-expert": "SimpleAgent", "distill-epochs": 5000,
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 300
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 4, "num-steps": 150,
#      "distill-expert": "SimpleAgent", "distill-epochs": 5000,
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 150
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "num-steps": 150,
#      "distill-expert": "SimpleAgent", "distill-epochs": 5000,
#     }, "pmanDSSimp"
# )
# train_ppo_job( # This job will have batch size of 400
#     {"num-processes": 8, "run-name": "gma99", "how-train": "simple", "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".99", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 4, "num-steps": 200,
#      "distill-expert": "SimpleAgent", "distill-epochs": 5000,
#     }, "pmanDSSimp"
# )


### These are meant to keep going on one of the earlier models that seemed to hit a wall wrt LR.
# train_dagger_job(
#     {"num-processes": 8, "run-name": "cont", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 25,
#      "minibatch-size": 2000, "save-interval": 25, "lr": 0.001, "num-steps-eval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-dagg-gma995.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb2000.ne15.prob0.5.nopt1.epoch290.steps232800.seed1.pt"
#  }
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "cont", "how-train": "dagger", "num-episodes-dagger": 15, "log-interval": 25,
#      "minibatch-size": 2000, "save-interval": 25, "lr": 0.0005, "num-steps-eval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#      "use-value-loss": "", "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-dagg-gma995.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb2000.ne15.prob0.5.nopt1.epoch290.steps232800.seed1.pt"
#  }
# )


### These are meant to see if we can distill the 1hot SImple Agent into the PPO while playing Team. These all have a bigger batch size.
### These did not work but did pose some questions. Namely, the 3e-4 learning rate ones
### seem to be learning something. We are goign to try increasing the KL factor a hell of a lot
### and seeing if that helps.
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"    
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v3", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"    
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"    
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "a", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.001,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"    
# )


### These try to adjust for what we learned above with an attempt to bump up the KL.
### 
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "b", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "b", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "b", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0005, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "b", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0005, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmanDSS"
# )


### These are hte same as above (modulo the lr) but repeated now that we have the new team reward in place.
### 
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "coop", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmancoop"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "coop", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000, "use-gae": "",
#     }, "pmancoop"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "coop", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#     }, "pmancoop"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "coop", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmall", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#     }, "pmancoop"
# )


### These are again trying to initialize from a dagger trained agent.
### 
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "initexuvl", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000, "use-gae": "",
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, 
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000, "use-gae": "",
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, 
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, 
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )


# ### These are like the above but additioanlyl distills from SimpleAgent
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "initexuvl", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000, "use-gae": "",
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000, "use-gae": "",
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "SimpleAgent", "distill-epochs": 10000,
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )


# ### These are like the above but additioanlyl distills from DaggerAgent
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "initexuvl", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000, "use-gae": "",
#      "distill-expert": "DaggerAgent", "distill-epochs": 10000,
#      "distill-target": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt",
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000, "use-gae": "",
#      "distill-expert": "DaggerAgent", "distill-epochs": 10000,
#      "distill-target": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt",
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 400
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 100, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "DaggerAgent", "distill-epochs": 10000,
#      "distill-target": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt",
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )
# train_ppo_job( # Batch size of 800
#     {"num-processes": 8, "run-name": "init", "how-train": "simple", "num-steps": 200, "log-interval": 100,
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamShort-v0", "gamma": ".995", "lr": 0.0003, "init-kl-factor": 10.0,
#      "model-str": "PommeCNNPolicySmaller", "num-mini-batch": 2, "half-lr-epochs": 5000,
#      "distill-expert": "DaggerAgent", "distill-epochs": 10000,
#      "distill-target": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt",
#      "saved-paths": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-lolexuvl.dagger.PommeTeamShort-v0.PommeCNNPolicySmaller.nc256.lr0.005.mb275.ne25.prob0.5.nopt1.epoch900.steps720800.seed1.pt"
#     }, "pmaninit"
# )


### These are trying to do homogenous training on EasyEnv but distilling from SimpleAgent
# These didn't work. They all went to random.
# train_ppo_job({ # Batch size of 400
#     "num-processes": 8, "run-name": "homoeasy", "how-train": "homogenous", 
#     "log-interval": 150, "lr": 0.0003, "num-steps": 100,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 10000, "distill-expert": "SimpleAgent",
#     "config": "PommeTeamEasy-v0", "eval-mode": "homogenous", "gamma": ".995",
#     "num-battles-eval": 50, "num-mini-batch": 2, "model-str": "PommeCNNPolicySmaller",
#     "half-lr-epochs": 5000, "use-gae": ""
# }, "pmanhomo")
# train_ppo_job({ # Batch size of 600
#     "num-processes": 8, "run-name": "homoeasy", "how-train": "homogenous", 
#     "log-interval": 150, "lr": 0.0003, "num-steps": 150,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 10000, "distill-expert": "SimpleAgent",
#     "config": "PommeTeamEasy-v0", "eval-mode": "homogenous", "gamma": ".995",
#     "num-battles-eval": 50, "num-mini-batch": 2, "model-str": "PommeCNNPolicySmaller",
#     "half-lr-epochs": 5000, "use-gae": ""
# }, "pmanhomo")
# train_ppo_job({ # Batch size of 400
#     "num-processes": 8, "run-name": "homoeasy", "how-train": "homogenous", 
#     "log-interval": 150, "lr": 0.0003, "num-steps": 100,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 10000, "distill-expert": "SimpleAgent",
#     "config": "PommeTeamEasy-v0", "eval-mode": "homogenous", "gamma": ".995",
#     "num-battles-eval": 50, "num-mini-batch": 2, "model-str": "PommeCNNPolicySmall",
#     "half-lr-epochs": 5000, "use-gae": ""
# }, "pmanhomo")
# train_ppo_job({ # Batch size of 600
#     "num-processes": 8, "run-name": "homoeasy", "how-train": "homogenous", 
#     "log-interval": 150, "lr": 0.0003, "num-steps": 150,
#     "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/",
#     "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/",
#     "distill-epochs": 10000, "distill-expert": "SimpleAgent",
#     "config": "PommeTeamEasy-v0", "eval-mode": "homogenous", "gamma": ".995",
#     "num-battles-eval": 50, "num-mini-batch": 2, "model-str": "PommeCNNPolicySmall",
#     "half-lr-epochs": 5000, "use-gae": ""
# }, "pmanhomo")


### Dagger agents on the easy env.
# These had good results in the tensorboard but were terrible when I ran eval.
# For example:
# CUDA_VISIBLE_DEVICES=0 python eval.py --eval-targets
# ppo::/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-easydag.dagger.PommeTeamEasy-v0.PommeCNNPolicySmall.nc256.lr0.003.mb275.ne20.prob0.5.nopt1.epoch1050.steps840800.seed1.pt
# --eval-opponents simple::null,simple::null --num-battles-eval 100 --config PommeTeamEasy-v0
# --cuda-device 0 --eval-mode team-simple --model-str PommeCNNPolicySmall
# had a 37% success rate in the TB but yielded 25 / 25 / 60 for w/l/t.
# CUDA_VISIBLE_DEVICES=0 python eval.py --eval-targets
# ppo::/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/agent0-dagger-easyexuvl.dagger.PommeTeamEasy-v0.PommeCNNPolicySmaller.nc256.lr0.003.mb275.ne20.prob0.5.nopt1.epoch900.steps720800.seed1.pt
# --eval-opponents simple::null,simple::null --num-battles-eval 100 --config PommeTeamEasy-v0
# --cuda-device 0 --eval-mode team-simple --model-str PommeCNNPolicySmaller
# had a 37% success rate in the TB but yielded 24 / 25 / 61. 
# train_dagger_job(
#     {"num-processes": 8, "run-name": "easyexuvl", "how-train": "dagger", "num-episodes-dagger": 20,
#      "log-interval": 50, "minibatch-size": 275, "save-interval": 50, "lr": 0.003, "num-steps-eval": 100, 
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamEasy-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "easyexuvl"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "easydag", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.003, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamEasy-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmaller",
#  }, "easydag"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "easyexuvl", "how-train": "dagger", "num-episodes-dagger": 20,
#      "log-interval": 50, "minibatch-size": 275, "save-interval": 50, "lr": 0.003, "num-steps-eval": 100, 
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamEasy-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#  }, "easyexuvl"
# )
# train_dagger_job(
#     {"num-processes": 8, "run-name": "easydag", "how-train": "dagger", "num-episodes-dagger": 20, "log-interval": 50,
#      "minibatch-size": 275, "save-interval": 50, "lr": 0.003, "num-steps-eval": 100, "use-value-loss": "",
#      "log-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/logs/", 
#      "save-dir": "/misc/kcgscratch1/ChoGroup/resnick/selfplayground/models/", 
#      "config": "PommeTeamEasy-v0", "gamma": ".995", "expert-prob": 0.5, "model-str": "PommeCNNPolicySmall",
#  }, "easydag"
# )


### FB Cluster Runs: These distill simple agent into PPO on the 1000 FFA dataset. Can we overfit?
# It appears that at least two of these did pretty darn well. The signatures for those:
# Both use gae, both use .99, and both have an LR of less than 1e-3
# 
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".99",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0001,
#     "use-gae": "", "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".99",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0001,
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".99",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .001,
#     "use-gae": "", "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".99",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .001,
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".99",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0003,
#     "use-gae": "", "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".99",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0003,
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".95",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0001,
#     "use-gae": "", "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".95",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0001,
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".95",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .001,
#     "use-gae": "", "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".95",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .001,
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".95",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0003,
#     "use-gae": "", "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)
# train_ppo_job({
#     "num-processes": 25, "run-name": "dstuni21", "how-train": "simple", 
#     "log-interval": 1000, "num-steps": 200,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "distill-epochs": 5000, "distill-expert": "SimpleAgent",
#     "config": "PommeFFAEasy-v0", "gamma": ".95",
#     "num-battles-eval": 100, "model-str": "PommeCNNPolicySmall", "lr": .0003,
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21"
# }, "pmanuni21", is_fb=True)


### This is a follow up to the experiments above
# It's a cartesian product of:
# {5000 distill, 2000 distill, no distill}, {LR of 7e-4, 3e-4}, and gamma of {.99, .995}
# except that the 5000 distill does not use the do {5000, 3e-4, .99} because it's already
# accounted for in the above.
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21", "use-gae": ""
# }
# counter = 0
# for learning_rate in [7e-4, 3e-4]:
#     for gamma in [.99, .995]:
#         for distill in [0, 2000, 5000]:
#             if distill == 5000 and gamma == .99 and learning_rate == 3e-4:
#                 continue
            
#             if distill:
#                 run_name = "pmansmpdst"
#                 job["distill-epochs"] = distill
#                 job["distill-expert"] = "SimpleAgent"
#             else:
#                 run_name = "pmansmp"

#             job["run-name"] = run_name + "-%d" % counter
#             job["gamma"] = gamma
#             job["lr"] = learning_rate
#             train_ppo_job(job, "pmansmp-%d" % counter, is_fb=True)
#             counter += 1


### These are the distill 0s from above. I fucked up and ran them incorrectly, so redoing here.
# job = {
#     "num-processes": 40, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform21", "use-gae": ""
# }
# counter = 0
# for learning_rate in [7e-4, 3e-4]:
#     for gamma in [.99, .995]:
#         for distill in [0]:
#             run_name = "pmansmp"
#             j = {k:v for k,v in job.items()}
#             j["run-name"] = run_name + "-%d" % counter
#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             train_ppo_job(j, "pmansmp-%d" % counter, is_fb=True)
#             counter += 1


### This is a further follow up to the experiments above using a bigger number of games (10000)
# It's a cartesian product of {5000 distill, 2000 distill, no distill}, {LR of 7e-4, 3e-4, 1e-4}, and gamma of {.99, .995}
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "state-directory": os.path.join(directory, "ffaeasy-10k-s100"),
#     "state-directory-distribution": "uniform21", "use-gae": ""
# }
# counter = 0
# for learning_rate in [7e-4, 3e-4, 1e-4]:
#     for gamma in [.99, .995]:
#         for distill in [0, 2000, 5000]:
#             j = {k:v for k,v in job.items()}
#             j["run-name"] = "pman10k-%d" % counter
#             if distill:
#                 j["distill-epochs"] = distill
#                 j["distill-expert"] = "SimpleAgent"

#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


### This is a follow up to the experiments two above using a longer uniform of 33.
# It's a cartesian product of {5000 distill, 2000 distill, no distill}, {LR of 7e-4, 3e-4, 1e-4}, and gamma of {.99, .995}
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "state-directory": os.path.join(directory, "ffaeasyv0-seed1"),
#     "state-directory-distribution": "uniform33", "use-gae": ""
# }
# counter = 0
# for learning_rate in [7e-4, 3e-4, 1e-4]:
#     for gamma in [.99, .995]:
#         for distill in [0, 2000, 5000]:
#             j = {k:v for k,v in job.items()}
#             j["run-name"] = "pman1k33-%d" % counter
#             if distill:
#                 j["distill-epochs"] = distill
#                 j["distill-expert"] = "SimpleAgent"

#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


### These are testing out the 8x8 agent to see if maybe PPO can work on that,
# possibly with a classification loss. These worked Really well!!!
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFA8x8-v0", "board-size": 8,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     # "eval-mode": "ffa-curriculum"
# }
# counter = 0
# for learning_rate in [3e-4, 1e-4]:
#     for gamma in [.99, .995]:
#         for distill in [0, 2500]:
#             j = {k:v for k,v in job.items()}
#             j["run-name"] = "pman8x8-%d" % counter
#             if distill:
#                 j["distill-epochs"] = distill
#                 j["distill-expert"] = "SimpleAgent"

#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


### More uniform experiments, this time uniform66 and uniformAdapt with 10k.
# 66 killed it! uniformAdapt I fucked upa nd am rerunning (see below)
# Cartesian product of {3000 distill, no distill}, {LR of 1e-4, 3e-5} and gamma of {.99, .995}
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffaeasy-10k-s100"),
# }
# counter = 0
# for learning_rate in [1e-4, 3e-5]:
#     for gamma in [.99, .995]:
#         for distill in [0, 3000]:
#             for (name, distro) in [("u66", "uniform66")]:
#                 j = {k:v for k,v in job.items()}
#                 j["run-name"] = "pman%s-%d" % (name, counter)
#                 j["state-directory-distribution"] = distro
#                 if distill:
#                     j["distill-epochs"] = distill
#                     j["distill-expert"] = "SimpleAgent"
#                 j["gamma"] = gamma
#                 j["lr"] = learning_rate
#                 train_ppo_job(j, j["run-name"], is_fb=True)
#                 counter += 1


### These are anneal bomb reward models.
# These didnt' work very well. They worked slightly better than the origianl, but still not
# well enough.
# Cartesian product of {3000 distill, no distill}, {LR of 1e-4, 3e-5} and gamma of {.99, .995}
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
# }
# counter = 0
# for learning_rate in [3e-4, 1e-4, 3e-5]:
#     for gamma in [.99, .995]:
#         for distill in [0, 3000]:
#             for anneal_bomb_penalty_epochs in [100, 1000, 5000]:
#                 j = {k:v for k,v in job.items()}
#                 j["run-name"] = "pmanABPE-%d" % counter
#                 if distill:
#                     j["distill-epochs"] = distill
#                     j["distill-expert"] = "SimpleAgent"
#                 j["gamma"] = gamma
#                 j["lr"] = learning_rate
#                 j["anneal-bomb-penalty-epochs"] = anneal_bomb_penalty_epochs
#                 train_ppo_job(j, j["run-name"], is_fb=True)
#                 counter += 1


### More uniform experiments, this time uniformAdapt and uniformScheduleA with 10k.
# Cartesian product of {3000 distill, no distill}, {LR of 1e-4, 6e-5} and gamma of {.99, .995}
# NOTE: The uniformAdpt on these use running_success_max_len=40. That's a lot (800 epochs).
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffaeasy-10k-s100"),
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.99, .995]:
#         for distill in [0, 3000]:
#             for (name, distro) in [("uSchA", "uniformScheduleA"), ("uAdpt", "uniformAdapt")]:
#                 for num_processes in [25, 50]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman%s-%d" % (name, counter)
#                     j["num-processes"] = num_processes
#                     j["state-directory-distribution"] = distro
#                     if distill:
#                         j["distill-epochs"] = distill
#                         j["distill-expert"] = "SimpleAgent"
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### These are the uniformAdapt as above but with ~200 epochs running_success_max_len=10, so roughly 4x faster. Also only running one gamma and no distill beause that seems to be ok.
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffaeasy-10k-s100"),
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.99]:
#         for distill in [0]:
#             for (name, distro) in [("uAdpt10", "uniformAdapt")]:
#                 for num_processes in [25, 50]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman%s-%d" % (name, counter)
#                     j["num-processes"] = num_processes
#                     j["state-directory-distribution"] = distro
#                     if distill:
#                         j["distill-epochs"] = distill
#                         j["distill-expert"] = "SimpleAgent"
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
                    

### This is a uniform66 test to see if we can run higher processor numbers (corresponding lower numsteps)
# It worked and is arguably better because it's faster.
# job = {
#     "num-processes": 50, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffaeasy-10k-s100"),
# }
# counter = 0
# for learning_rate in [1e-4]:
#     for gamma in [.99, .995]:
#         for distill in [0, 3000]:
#             for (name, distro) in [("u66", "uniform66")]:
#                 j = {k:v for k,v in job.items()}
#                 j["run-name"] = "pman%s-%d-hnp" % (name, counter)
#                 j["state-directory-distribution"] = distro
#                 if distill:
#                     j["distill-epochs"] = distill
#                     j["distill-expert"] = "SimpleAgent"
#                 j["gamma"] = gamma
#                 j["lr"] = learning_rate
#                 train_ppo_job(j, j["run-name"], is_fb=True)
#                 counter += 1


# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffaeasy-10k-s100"),
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.99, .995]:
#         for distill in [0]:
#             for (name, distro) in [("uBndsA", "uniformBoundsA"), ("uBndsB", "uniformBoundsB")]:
#                 for num_processes in [25, 50]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman%s-%d" % (name, counter)
#                     j["num-processes"] = num_processes
#                     j["state-directory-distribution"] = distro
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


# These are testing out the 8x8 agent to see if maybe PPO can work on that,
# possibly with a classification loss. These are on the no sjull set up though.
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFA8x8-v0", "board-size": 8,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     # "eval-mode": "ffa-curriculum"
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.99, .995]:
#         for distill in [0, 3000]:
#             j = {k:v for k,v in job.items()}
#             j["run-name"] = "pman8x8nsk-%d" % counter
#             if distill:
#                 j["run-name"] += "-dst"
#                 j["distill-epochs"] = distill
#                 j["distill-expert"] = "SimpleAgent"
#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


### These are homogenous jobs on 8x8.
# job = {
#     "num-processes": 30, "how-train": "homogenous", "eval-mode": "homogenous",
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeTeam8x8-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "use-gae": "", "board-size": 8,
# }
# counter = 0
# for learning_rate in [3e-4, 1e-4, 6e-5]:
#     for gamma in [.99, .995]:
#         j = {k:v for k,v in job.items()}        
#         run_name = "pmanhom8x8"
#         j["run-name"] = run_name + "-%d" % counter
#         j["gamma"] = gamma
#         j["lr"] = learning_rate
#         train_ppo_job(j, j["run-name"], is_fb=True)
#         counter += 1


### Homog similar to above but using distill as well.
# job = {
#     "num-processes": 30, "how-train": "homogenous", "eval-mode": "homogenous",
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeTeam8x8-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "use-gae": "", "board-size": 8, "distill-expert": "SimpleAgent"
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.995, .999]:
#         for distill in [2500, 5000]:
#             j = {k:v for k,v in job.items()}        
#             run_name = "pmanhomdst8x8"
#             j["run-name"] = run_name + "-%d" % counter
#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             j["distill-epochs"] = distill
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


### These are being run with a very small dataset of just 4 things.
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFAEasy-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffacompetition4-s100/train")
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.995]:
#         for distill in [0, 3000]:
#             for (name, distro) in [("uSchA", "uniformScheduleA"),
#                                    ("uAdpt", "uniformAdapt"),
#                                    ("uBnA", "uniformBoundsA"),                                   
#                                    ("uBnB", "uniformBoundsB")]:
#                 for num_processes in [50]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman4%s-%d" % (name, counter)
#                     j["num-processes"] = num_processes
#                     j["state-directory-distribution"] = distro
#                     if distill:
#                         j["distill-epochs"] = distill
#                         j["distill-expert"] = "SimpleAgent"
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### These are redos for the new dataset.
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffacompetition11k-s10000/train")
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.995]:
#         for distill in [0, 3000]:
#             for (name, distro) in [("uSchA", "uniformScheduleA"),
#                                    ("uAdpt", "uniformAdapt"),
#                                    ("uBnA", "uniformBoundsA"),                                   
#                                    ("uBnB", "uniformBoundsB")]:
#                 for num_processes in [50]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman11kTr-%s-%d" % (name, counter)
#                     j["num-processes"] = num_processes
#                     j["state-directory-distribution"] = distro
#                     if distill:
#                         j["distill-epochs"] = distill
#                         j["distill-expert"] = "SimpleAgent"
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
            



### Homog with reward shaping.
# These allll sucked.
# job = {
#     "num-processes": 50, "how-train": "homogenous", "eval-mode": "homogenous",
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeTeam8x8-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "use-gae": "", "board-size": 8
# }
# counter = 0
# for learning_rate in [1e-4]:
#     for gamma in [.995, 1.]:
#         for distill in [0, 3000]:
#             for bomb_reward in [0.0, 0.001, 0.01]:
#                 for step_loss in [0.0, -0.001, -0.01]:
#                     for begin_selfbombing_epoch in [0, 100, 500]:
#                         if bomb_reward == 0 and step_loss == 0 and begin_selfbombing_epoch == 0:
#                             continue

#                         j = {k:v for k,v in job.items()}
#                         j["run-name"] = "pmhom8x8"

#                         if begin_selfbombing_epoch > 0:
#                             j["run-name"] += "-bsbe%d" % begin_selfbombing_epoch
#                             j["begin-selfbombing-epoch"] = begin_selfbombing_epoch

#                         if bomb_reward:
#                             j["run-name"] += "-br%d" % int(1000*bomb_reward)
#                             j["bomb-reward"] = bomb_reward

#                         if step_loss:
#                             j["run-name"] += "-st%d" % int(-1000*step_loss)
#                             j["step-loss"] = step_loss

#                         if distill:
#                             j["distill-expert"] = "SimpleAgent"
#                             j["distill-epochs"] = distill

#                         j["run-name"] = j["run-name"] + "-%d" % counter
#                         j["gamma"] = gamma
#                         j["lr"] = learning_rate
#                         train_ppo_job(j, j["run-name"], is_fb=True)
#                         counter += 1


# What happens if we do homogenous with a large begin_slefbombing_epoch?
# job = {
#     "num-processes": 50, "how-train": "homogenous", "eval-mode": "homogenous",
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeTeam8x8-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "use-gae": "", "board-size": 8
# }
# counter = 0
# for learning_rate in [1e-4]:
#     for gamma in [.995, 1.]:
#         for distill in [0, 3000]:
#             for begin_selfbombing_epoch in [10000]:
#                 j = {k:v for k,v in job.items()}
#                 j["run-name"] = "pmhom8x8"
#                 j["run-name"] += "-bsbe%d" % begin_selfbombing_epoch
#                 j["begin-selfbombing-epoch"] = begin_selfbombing_epoch
#                 if distill:
#                     j["distill-expert"] = "SimpleAgent"
#                     j["distill-epochs"] = distill
                    
#                 j["run-name"] = j["run-name"] + "-%d" % counter
#                 j["gamma"] = gamma
#                 j["lr"] = learning_rate
#                 train_ppo_job(j, j["run-name"], is_fb=True)
#                 counter += 1
                        

# ### These are being run with a very small dataset of just 4 things.
# # Testing what happens if we use a large begin_selfbombing_epoch.
# # They had the same effect as before, namely that nothiung interesting happened
# # after the selfbombing was allowed again.
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0",
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffacompetition4-s100/train"),
#     "num-processes": 50,
# }
# counter = 0
# for learning_rate in [1e-4]:
#     for gamma in [.995, 1]:
#         for distill in [0, 3000]:
#             for (name, distro) in [
#                     ("uAdpt", "uniformAdapt"),
#                     ("uSchA", "uniformScheduleB"),
#             ]:
#                 for begin_selfbombing_epoch in [6000]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman4%s-lgbsbe%d-%d" % (
#                         name, begin_selfbombing_epoch, counter)
#                     j["begin-selfbombing-epoch"] = begin_selfbombing_epoch
#                     j["state-directory-distribution"] = distro
#                     if distill:
#                         j["distill-epochs"] = distill
#                         j["distill-expert"] = "SimpleAgent"
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### These are doing 8x8 simple ffa with the begin_selfbombing_epoch.
# job = {
#     "num-processes": 50, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFA8x8-v0", "board-size": 8,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     # "eval-mode": "ffa-curriculum"
# }
# counter = 0
# for learning_rate in [1e-4]:
#     for gamma in [.995, 1.]:
#         for distill in [0, 3000]:
#             for begin_selfbombing_epoch in [1000, 10000]:
#                 j = {k:v for k,v in job.items()}
#                 j["run-name"] = "pman8x8nsk-lgbsbe%d-%d" % (
#                     begin_selfbombing_epoch, counter)
#                 j["begin-selfbombing-epoch"] = begin_selfbombing_epoch
#                 if distill:
#                     j["run-name"] += "-dst"
#                     j["distill-epochs"] = distill
#                     j["distill-expert"] = "SimpleAgent"
#                 j["gamma"] = gamma
#                 j["lr"] = learning_rate
#                 train_ppo_job(j, j["run-name"], is_fb=True)
#                 counter += 1


### Homog but this time against the simple agent as the starting challenger.
# TODO: THESE DIDNT START ... WHy not??
# job = {
#     "num-processes": 30, "how-train": "homogenous", "eval-mode": "homogenous",
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeTeam8x8-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall",
#     "use-gae": "", "board-size": 8, 
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.995, .999]:
#         for distill in [0, 2500, 5000]:
#             run_name = "pmanhom8smpst"
#             j = {k:v for k,v in job.items()}
#             if distill:
#                 j["distill-expert"] = "SimpleAgent"
#                 j["distill-epochs"] = distill
#                 run_name += "dst"
#             j["run-name"] = run_name + "-%d" % counter
#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


# TODO
### These are attempting to do reward shaping. We are using the small dataset
### because then we'll get results faster.
### Yeah these didn't work. They achieeved full success, but it wasn't sufficient
### to get the agents to learn another approach other than "don't bomb."
#### WAIT< BUT NONE OF THESE USED STEP LSOS OR BOMB REWARD DUMBASS
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "num-battles-eval": 100,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffacompetition4-s100/train"),
#     "num-processes": 50,
# }
# counter = 0
# for learning_rate in [1e-4]:
#     for gamma in [.995]:
#         for distill in [0, 3000]:
#             for (name, distro) in [
#                     ("uAdpt", "uniformAdapt"),
#                     ("uBnA", "uniformBoundsA")
#             ]:
#                 for bomb_reward in [0.0, 0.05, 0.1]:
#                     for step_loss in [0.0, -0.05, -0.1]:
#                         if bomb_reward == 0.0 and step_loss == 0.0:
#                             continue

#                         j = {k:v for k,v in job.items()}
#                         j["run-name"] = "pman4"

#                         if bomb_reward:
#                             j["run-name"] += "br%d" % int(100*bomb_reward)
#                         if step_loss:
#                             j["run-name"] += "st%d" % int(100*step_loss)

#                         if distill:
#                             j["distill-epochs"] = distill
#                             j["distill-expert"] = "SimpleAgent"
#                             j["run-name"] += "-dst"

#                         j["state-directory-distribution"] = distro
#                         j["run-name"] += "-%s-%d" % (name, counter)
#                         j["gamma"] = gamma
#                         j["lr"] = learning_rate
#                         train_ppo_job(j, j["run-name"], is_fb=True)
#                         counter += 1
# # These are the same as above but with 8x8.
# job = {
#     "num-processes": 50, "how-train": "simple", 
#     "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFA8x8-v0", "board-size": 8,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     # "eval-mode": "ffa-curriculum"
# }
# counter = 0
# for learning_rate in [1e-4]:
#     for gamma in [.995]:
#         for distill in [0, 3000]:
#             for bomb_reward in [0.0, 0.05, 0.1]:
#                 for step_loss in [0.0, -0.05, -0.1]:
#                     if bomb_reward == 0.0 and step_loss == 0.0:
#                         continue
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman8x8"
#                     if bomb_reward:
#                         j["run-name"] += "br%d" % int(100*bomb_reward)
#                     if step_loss:
#                         j["run-name"] += "st%d" % int(100*step_loss)

#                     j["state-directory-distribution"] = distro
#                     if distill:
#                         j["distill-epochs"] = distill
#                         j["distill-expert"] = "SimpleAgent"
#                         j["run-name"] += "-dst"

#                     j["run-name"] += "-%s-%d" % (name, counter)
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### These are repeating teh 8x8 test but with complex agents because they don't
### kill themsevles.
# job = {
#     "num-processes": 25, "how-train": "simple", 
#     "log-interval": 1000,  "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFA8x8-v0", "board-size": 8,
#     "model-str": "PommeCNNPolicySmall", "use-gae": "",
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.99, .995, 1.]:
#         for distill in [0, 2500, 5000]:
#             j = {k:v for k,v in job.items()}
#             j["run-name"] = "pmancmplx8x8-%d" % counter
#             if distill:
#                 j["distill-epochs"] = distill
#                 j["distill-expert"] = "ComplexAgent"

#             j["gamma"] = gamma
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


### These are being run with a very small dataset of just 4 things and with ComplexAgent.
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "state-directory": os.path.join(directory, "ffacompetition4-s100-complex/train")
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for gamma in [.995, 1.]:
#         for distill in [0, 2500, 5000]:
#             for (name, distro) in [("uSchA", "uniformScheduleA"),
#                                    ("uAdpt", "uniformAdapt"),
#                                    ("uBnA", "uniformBoundsA")]:
#                 for num_processes in [50]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "pman4cmplx%s-%d" % (name, counter)
#                     j["num-processes"] = num_processes
#                     j["state-directory-distribution"] = distro
#                     if distill:
#                         j["distill-epochs"] = distill
#                         j["distill-expert"] = "ComplexAgent"
#                     j["gamma"] = gamma
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
            

### These mostly worked, however they didnt end up getting to where we wanted wrt
### going further back in time because they hit some odd bugs.
### A note is that while the reduced schedules did not seem to work for 100, they did
### seem to work for 4.
# job = {
#     "how-train": "simple",  "log-interval": 1000,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5, 3e-5]:
#     for (name, distro) in [
#             ("uSchB", "uniformScheduleB"),
#             ("uSchC", "uniformScheduleC"),
#             ("uSchD", "uniformScheduleD"),
#             ("uSchE", "uniformScheduleE"),
#             ("uSchF", "uniformScheduleF"),
#             ("uSchG", "uniformScheduleG"),                        
#             ("uBnD", "uniformBoundsD"),
#             ("uBnE", "uniformBoundsE"),
#             ("uBnA", "uniformBoundsA")                               
#     ]:
#         for numgames in [100, 4]:
#             j = {k:v for k,v in job.items()}
#             j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#             j["run-name"] = "pman%dcmplx2%s-%d" % (numgames, name, counter)
#             j["state-directory-distribution"] = distro
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1
                

### Similar to the above, but for 4. To be honest, this is basically the experiemnt right here.
### We are asking: Can we learn better from the back than from the front?
# These are still going. They are doing well but we want to go faster and hone in on the backend.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5, 3e-5]:
#     for (name, distro) in [
#             ("uSchC", "uniformScheduleC"), #500
#             ("uSchB", "uniformScheduleB"), #1000
#             ("uSchF", "uniformScheduleF"), #2000
#             ("uBnF", "uniformBoundsF"), #500
#             ("uBnB", "uniformBoundsB"), #1000
#             ("uBnE", "uniformBoundsE"), #2000
#             ("genesis", "genesis"),
#             ("uFwdA", "uniformForwardA"), #250
#             ("uFwdB", "uniformForwardB"), #500
#             ("uFwdC", "uniformForwardC"), #1000
#             ("uAll", "uniform"), #all random.
#             ("ubtst", "uniformBoundsBTst"),
#     ]:
#         for numgames in [4]:
#             if name == "ubtst" and learning_rate in [1e-4, 6e-5]:
#                 continue
            
#             j = {k:v for k,v in job.items()}
#             j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#             j["run-name"] = "cmplxagn-%d-%s-%d" % (numgames, name, counter)
#             j["state-directory-distribution"] = distro
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


### LSTM runs with distribution: 16
# job = {
#     "num-processes": 50, "how-train": "simple",
#     "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "gamma": "1.0", "use-gae": "",
#     "model-str": "PommeCNNPolicySmall",
#     "recurrent-policy": "", "eval-mode": "ffa", "num-stack": 1,
#     "config": "PommeFFACompetition-v0"
# }
# counter = 0
# for learning_rate in [1e-4, 3e-5]:
#     for (name, distro) in [
#             ("exdistr", None),
#             ("genesis", "genesis"), # always starts at step 0 from replays.
#             ("uSchC", "uniformScheduleC"), #500
#             ("uSchB", "uniformScheduleB"), #1000
#             ("uSchF", "uniformScheduleF"), #2000
#             ("uBnF", "uniformBoundsF"), #500
#             ("uBnB", "uniformBoundsB"), #1000
#             ("uBnE", "uniformBoundsE"), #2000
#     ]:
#         if distro:
#             j["state-directory"] = os.path.join(directory, "ffacompetition4-s100-complex/train")
#             j["state-directory-distribution"] = distro

#         j = {k:v for k,v in job.items()}
#         j["run-name"] = "lstm-%s-pman%d" % (name, counter)
#         j["lr"] = learning_rate
#         train_ppo_job(j, j["run-name"], is_fb=True)
#         counter += 1


# ### LSTM with 8x8: 2
# job = {
#     "num-processes": 50, "how-train": "simple",
#     "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"),
#     "save-dir": os.path.join(directory, "models"),
#     "gamma": "1.0", "use-gae": "",
#     "model-str": "PommeCNNPolicySmall",
#     "recurrent-policy": "", "eval-mode": "ffa", "num-stack": 1,
#     "config": "PommeFFA8x8-v0", "board-size": 8,
# }
# counter = 0
# for learning_rate in [1e-4, 3e-5]:
#     j = {k:v for k,v in job.items()}
#     j["run-name"] = "lstm8x8-pman%d" % (counter)
#     j["lr"] = learning_rate
#     train_ppo_job(j, j["run-name"], is_fb=True)
#     counter += 1


### Doing the above (non-lstm jobs) but having honed in on the backend more.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5, 3e-5]:
#     for (name, distro) in [
#             ("setBnA", "setBoundsA"),
#             ("setBnB", "setBoundsB"),
#     ]:
#         for numgames in [4]:
#             j = {k:v for k,v in job.items()}
#             j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#             j["run-name"] = "cmplxfstr-%d-%s-%d" % (numgames, name, counter)
#             j["state-directory-distribution"] = distro
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


job = {
    "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
    "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
    "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
    "num-processes": 50, "gamma": 1.0,
}
counter = 0
for learning_rate in [1e-4, 6e-5, 3e-5]:
    for (name, distro) in [
            ("setBnC", "setBoundsC"),
            ("setBnD", "setBoundsD"),
    ]:
        for numgames in [4]:
            j = {k:v for k,v in job.items()}
            j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
            j["run-name"] = "cmplxfstr-%d-%s-%d" % (numgames, name, counter)
            j["state-directory-distribution"] = distro
            j["lr"] = learning_rate
            train_ppo_job(j, j["run-name"], is_fb=True)
            counter += 1
