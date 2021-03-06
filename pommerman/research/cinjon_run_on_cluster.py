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
    "max-aggregate-agent-states": "maxaggr",
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
    'use-both-places': 'ubp',
    'mix-frozen-complex': 'mfc',
    'adapt-threshold': 'adpt',
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
    s += "--mem=64000 --time=72:00:00 %s &" % os.path.join(
        slurm_scripts, jobnameattrs + ".slurm")
    os.system(s)


def train_dagger_job(flags, jobname=None, is_fb=False, partition="uninterrupted"):
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
            f.write("#SBATCH --partition=%s\n" % partition)
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
    s += "--mem=64000 --time=72:00:00 %s &" % os.path.join(
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
# saved_paths = {
#     "cmplxagn-4-uBnB-4": "agent0-cmplxagn-4-uBnB-4.simple.FFACompetition-v0.Small.nc256.lr0.0001.bs5120.ns103.gam1.0.seed1.gae.uniformBoundsBepoch4750.steps24462500.pt",
#     "cmplxagn-4-uBnB-15": "agent0-cmplxagn-4-uBnB-15.simple.FFACompetition-v0.Small.nc256.lr6e-05.bs5120.ns103.gam1.0.seed1.gae.uniformBoundsBepoch4500.steps23175000.pt",
# }
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
#             # ("ubtst", "uniformBoundsBTst"),
#     ]:
#         for numgames in [4]:
#             if counter not in [4, 15]:
#                 counter += 1
#                 continue
#             j = {k:v for k,v in job.items()}
#             j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#             j["run-name"] = "cmplxagn-%d-%s-%d" % (numgames, name, counter)
#             path = saved_paths.get(j["run-name"])
#             if not path:
#                 counter += 1
#                 continue
#             else:
#                 j["saved-paths"] = "/checkpoint/cinjon/selfplayground/models/%s" % path
#                 print("DOING SAVED PATH: ", j["saved-paths"])
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


# saved_paths = {
#     "cmplxfstr-4-setBnC-4": "agent0-cmplxfstr-4-setBnC-4.simple.FFACompetition-v0.Small.nc256.lr3e-05.bs5120.ns103.gam1.0.seed1.gae.setBoundsCepoch1250.steps6437500.pt",
#     "cmplxfstr-4-setBnC-2": "agent0-cmplxfstr-4-setBnC-2.simple.FFACompetition-v0.Small.nc256.lr6e-05.bs5120.ns103.gam1.0.seed1.gae.setBoundsCepoch1250.steps6437500.pt",
#     "cmplxfstr-4-setBnD-1": "agent0-cmplxfstr-4-setBnD-1.simple.FFACompetition-v0.Small.nc256.lr0.0001.bs5120.ns103.gam1.0.seed1.gae.setBoundsDepoch1250.steps6437500.pt",
#     "cmplxfstr-4-setBnF-3": "agent0-cmplxfstr-4-setBnF-3.simple.FFACompetition-v0.Small.nc256.lr6e-05.bs5120.ns103.gam1.0.seed1.gae.setBoundsFepoch1250.steps6437500.pt",
#     "cmplxfstr-4-setBnE-4": "agent0-cmplxfstr-4-setBnE-4.simple.FFACompetition-v0.Small.nc256.lr3e-05.bs5120.ns103.gam1.0.seed1.gae.setBoundsEepoch1250.steps6437500.pt",
#     "cmplxfstr-4-setBnD-5": "agent0-cmplxfstr-4-setBnD-5.simple.FFACompetition-v0.Small.nc256.lr3e-05.bs5120.ns103.gam1.0.seed1.gae.setBoundsDepoch1250.steps6437500.pt",
#     "cmplxfstr-4-setBnE-0": "agent0-cmplxfstr-4-setBnE-0.simple.FFACompetition-v0.Small.nc256.lr0.0001.bs5120.ns103.gam1.0.seed1.gae.setBoundsEepoch1250.steps6437500.pt",
#     "cmplxfstr-4-setBnF-1": "agent0-cmplxfstr-4-setBnF-1.simple.FFACompetition-v0.Small.nc256.lr0.0001.bs5120.ns103.gam1.0.seed1.gae.setBoundsFepoch1250.steps6437500.pt",
# }
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5, 3e-5]:
#     for (name, distro) in [
#             ("setBnC", "setBoundsC"),
#             ("setBnD", "setBoundsD"),
#     ]:
#         for numgames in [4]:
#             if counter not in [1, 2, 4, 5]:
#                 counter += 1
#                 continue
#             j = {k:v for k,v in job.items()}
#             j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#             j["run-name"] = "cmplxfstr-%d-%s-%d" % (numgames, name, counter)
#             path = saved_paths.get(j["run-name"])
#             if not path:
#                 counter += 1
#                 continue
#             else:
#                 j["saved-paths"] = "/checkpoint/cinjon/selfplayground/models/%s" % path
#                 print("DOING SAVED PATH: ", j["saved-paths"])
#             j["state-directory-distribution"] = distro
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1


# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5, 3e-5]:
#     for (name, distro) in [
#             # ("setBnC", "setBoundsC"),
#             # ("setBnD", "setBoundsD"),
#             ("setBnE", "setBoundsE"),
#             ("setBnF", "setBoundsF"),
#     ]:
#         for numgames in [4]:
#             if counter not in [0, 1, 3, 4]:
#                 counter += 1
#                 continue
#             j = {k:v for k,v in job.items()}
#             j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#             j["run-name"] = "cmplxfstr-%d-%s-%d" % (numgames, name, counter)
#             path = saved_paths.get(j["run-name"])
#             if not path:
#                 counter += 1
#                 continue
#             else:
#                 j["saved-paths"] = "/checkpoint/cinjon/selfplayground/models/%s" % path
#                 print("DOING SAVED PATH: ", j["saved-paths"])
#             j["state-directory-distribution"] = distro
#             j["lr"] = learning_rate
#             train_ppo_job(j, j["run-name"], is_fb=True)
#             counter += 1
            

### Test adding an item reward in.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for (name, distro) in [
#             ("setBnF", "setBoundsF"),
#             ("setBnD", "setBoundsD"),
#             ("uBnF", "uniformBoundsF"), #500
#             ("uBnB", "uniformBoundsB"), #1000
#     ]:
#         for numgames in [4, 100]:
#             for itemreward in [0, .03, .1]:
#                 if itemreward == 0 and numgames == 4:
#                     continue

#                 if counter not in [0, 36, 30, 10, 25]:
#                     counter += 1
#                     continue
                
#                 j = {k:v for k,v in job.items()}
#                 j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                 j["run-name"] = "cmplxitm%d-%s-%d" % (numgames, name, counter)
#                 if itemreward:
#                     j["item-reward"] = itemreward
#                 j["state-directory-distribution"] = distro
#                 j["lr"] = learning_rate
#                 train_ppo_job(j, j["run-name"], is_fb=True)
#                 counter += 1
            

### Same as aobve but including some other runs.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for (name, distro) in [
#             ("setBnF", "setBoundsF"),
#             ("setBnD", "setBoundsD"),
#             ("uBnF", "uniformBoundsF"), #500
#             ("uBnB", "uniformBoundsB"), #1000
#     ]:
#         for numgames in [4]:
#             for itemreward in [.01, 0]:
#                 if counter not in [4, 14, 5, 12, 8, 1, 3, 0, 2]:
#                     counter += 1
#                     continue
#                 j = {k:v for k,v in job.items()}
#                 j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                 j["run-name"] = "cmplxitm2-%d-%s-%d" % (numgames, name, counter)
#                 if itemreward:
#                     j["item-reward"] = itemreward
#                 j["state-directory-distribution"] = distro
#                 j["lr"] = learning_rate
#                 train_ppo_job(j, j["run-name"], is_fb=True)
#                 counter += 1


### Uninterrupted version of the above :/.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 250,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for (name, distro) in [
#             ("setBnF", "setBoundsF"),
#             ("setBnD", "setBoundsD"),
#             ("uBnF", "uniformBoundsF"), #500
#             ("uBnB", "uniformBoundsB"), #1000
#     ]:
#         for numgames in [4]:
#             for itemreward in [0, .01, .03, .1]:
#                 for seed in [2, 3]:
#                     j = {k:v for k,v in job.items()}
#                     j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                     j["run-name"] = "unint-cmplxitm%d-%s-%d" % (numgames, name, counter)
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### With the dumb batch size fix.
# NOTE: This is the SMALL BATCH SIZE / WARM START VERSION!!!
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 500,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 50, "gamma": 1.0,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for (name, distro) in [
#             ("setBnF", "setBoundsF"),
#             ("setBnD", "setBoundsD"),
#             ("uBnF", "uniformBoundsF"), #500
#             ("uBnB", "uniformBoundsB"), #1000
#     ]:
#         for numgames in [4]:
#             for itemreward in [0, .03, .1]:
#                 for seed in [2]:
#                     j = {k:v for k,v in job.items()}
#                     j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                     j["run-name"] = "untbsfx-%s-%d" % (name, counter)
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
                    

### These were promising but take hella long to run, so we want to expedite them and save them more frequently..
### NOTE: THIS IS THE BIG BATCH SIZE / WARM START
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [1e-4, 6e-5]:
#     for (name, distro) in [
#             ("setBnF", "setBoundsF"),
#             ("setBnD", "setBoundsD"),
#             ("uBnF", "uniformBoundsF"), #500
#             ("uBnB", "uniformBoundsB"), #1000
#             ("genesis", "genesis"),
#     ]:
#         for numgames in [4]:
#             for itemreward in [0, .1]:
#                 for seed in [1]:
#                     j = {k:v for k,v in job.items()}
#                     j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                     j["run-name"] = "bsfx-%s-%d" % (name, counter)
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### This is the main comparison between genesis and uBn.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-bsfx"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4, 1e-4]:
#     for (name, distro) in [
#             ("uBnG", "uniformBoundsG"), #50
#             ("uBnH", "uniformBoundsH"), #100
#             ("genesis", "genesis"),
#     ]:
#         for numgames in [4]:
#             for itemreward in [0, .03, .1]:
#                 for seed in [1]:
#                     if itemreward == 0 and name == "genesis":
#                         continue
#                     j = {k:v for k,v in job.items()}
#                     j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                     j["run-name"] = "bsfx-%s-%d" % (name, counter)
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### These are doing the above but for 100 games. Consequently, we had to apply longer bounds.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-bsfx100"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             ("uBnG", "uniformBoundsG"), #50
#             ("uBnJ", "uniformBoundsJ"), #75            
#             ("uBnH", "uniformBoundsH"), #100
#             ("uBnI", "uniformBoundsI"), #150
#             ("genesis", "genesis"),
#     ]:
#         for numgames in [100]:
#             for itemreward in [0, .1]:
#                 for seed in [1, 2]:
#                     if itemreward == 0 and name == "genesis":
#                         continue
#                     j = {k:v for k,v in job.items()}
#                     j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                     j["run-name"] = "100bds-%s-%d" % (name, counter)
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
                    
                    
### This is using the 2nd place agent as the expert. They were there to the end, but received a -1 at the end.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-bsfxusp"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000, "use-second-place": ""
# }
# counter = 0
# for learning_rate in [1e-4, 3e-4]:
#     for (name, distro) in [
#             ("uBnG", "uniformBoundsG"), #50
#             ("uBnK", "uniformBoundsK"), #40
#     ]:
#         for numgames in [4]:
#             for itemreward in [0, .1]:
#                 for seed in [1, 2]:
#                     j = {k:v for k,v in job.items()}
#                     j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex-2nd/train" % numgames)
#                     j["run-name"] = "bsfxusp-%s-%d" % (name, counter)
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### Same as two above with 100 games, but here we use an epoch change of 85.
# job = {
#     "how-train": "simple",  "log-interval": 2500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-bsfx100"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             ("uBnL", "uniformBoundsL"), #85
#     ]:
#         for numgames in [100]:
#             for itemreward in [0, .1]:
#                 for seed in [1, 2]:
#                     j = {k:v for k,v in job.items()}
#                     j["state-directory"] = os.path.join(directory, "ffacompetition%d-s100-complex/train" % numgames)
#                     j["run-name"] = "100bds-%s-%d" % (name, counter)
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### Use the rebuilt replays! Fucking lol.
# Um, these all failed before for a mysterious reason that I'm not aware of..
# Doing it again.
# job = {
#     "how-train": "simple",  "log-interval": 7500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-fxrp"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             ("uBnG", "uniformBoundsG"), #50
#             ("uBnJ", "uniformBoundsJ"), #75
#             ("uBnL", "uniformBoundsL"), #85
#             ("uBnH", "uniformBoundsH"), #100
#             ("genesis", "genesis"),
#     ]:
#         for numgames in [110, 5]:
#             if numgames == 5 and name in ["uBnH", "uBnL"]:
#                 # Skip distributions we know are too slow.
#                 continue

#             for itemreward in [0, .1]:
#                 for seed in [1, 2]:
#                     for use_second_place in [True, False]:
#                         if numgames == 110:
#                             runng = 100
#                         elif numgames == 5:
#                             runng = 4
#                         j = {k:v for k,v in job.items()}
#                         subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                         log_dir = os.path.join(directory, "logs-fx%d" % runng)
#                         save_dir = os.path.join(directory, "models-fx%d" % runng)
#                         run_name = "fx%d-%s-%d" % (runng, name, counter)
#                         if use_second_place:
#                             if numgames == 110:
#                                 # Skip because we didn't set up usp for 110
#                                 counter += 1
#                                 continue
#                             j["use-second-place"] = ""
#                             subdir += "-2nd"
#                             log_dir += "usp"
#                             save_dir += "usp"
#                             run_name += "usp"
                            
#                         j["state-directory"] = os.path.join(
#                             directory,
#                             "pomplays",
#                             subdir,
#                             "train")
#                         j["log-dir"] = log_dir
#                         j["save-dir"] = save_dir
#                         j["run-name"] = run_name
#                         if itemreward:
#                             j["item-reward"] = itemreward
#                         j["seed"] = seed
#                         j["state-directory-distribution"] = distro
#                         j["lr"] = learning_rate
#                         train_ppo_job(j, j["run-name"], is_fb=True)
#                         counter += 1



### Doing different seeds on the G models...
# job = {
#     "how-train": "simple",  "log-interval": 7500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-fxrp"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             # ("uBnG", "uniformBoundsG"), #50
#             ("uniform", "uniform"), #50            
#     ]:
#         for numgames in [5]:
#             for itemreward in [0, .1]:
#                 for seed in [3, 4, 5]:
#                     for use_second_place in [True, False]:
#                         runng = 4
#                         j = {k:v for k,v in job.items()}
#                         subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                         log_dir = os.path.join(directory, "logs-fx%d" % runng)
#                         save_dir = os.path.join(directory, "models-fx%d" % runng)
#                         run_name = "2fx%d-%s-%d" % (runng, name, counter)
                        
#                         if use_second_place:
#                             j["use-second-place"] = ""
#                             subdir += "-2nd"
#                             log_dir += "usp"
#                             save_dir += "usp"
#                             run_name += "usp"
                            
#                         j["state-directory"] = os.path.join(
#                             directory,
#                             "pomplays",
#                             subdir,
#                             "train")
#                         j["log-dir"] = log_dir
#                         j["save-dir"] = save_dir
#                         j["run-name"] = run_name
#                         if itemreward:
#                             j["item-reward"] = itemreward
#                         j["seed"] = seed
#                         j["state-directory-distribution"] = distro
#                         j["lr"] = learning_rate
#                         train_ppo_job(j, j["run-name"], is_fb=True)
#                         counter += 1
                        

### Doing different seeds on the 100 L models...
# job = {
#     "how-train": "simple",  "log-interval": 7500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-fxrp"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             # ("uBnL", "uniformBoundsL"), #85
#             ("uniform", "uniform"), #85            
#     ]:
#         for numgames in [110]:
#             for itemreward in [0, .1]:
#                 for seed in [3, 4, 5]:
#                     runng = 100
#                     j = {k:v for k,v in job.items()}
#                     subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                     log_dir = os.path.join(directory, "logs-fx%d" % runng)
#                     save_dir = os.path.join(directory, "models-fx%d" % runng)
#                     run_name = "2fx%d-%s-%d" % (runng, name, counter)
#                     j["state-directory"] = os.path.join(
#                         directory,
#                         "pomplays",
#                         subdir,
#                         "train")
#                     j["log-dir"] = log_dir
#                     j["save-dir"] = save_dir
#                     j["run-name"] = run_name
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


# Dagger, the 102400 ones didn't work because we ran out of memory... I'm going to have to restart these to be have more time...
# job = {
#     "num-processes": 60, "how-train": "dagger", "num-episodes-dagger": 10, "log-interval": 25,
#     "save-interval": 25, "num-steps-eval": 100,
#     "log-dir": os.path.join(directory, "dagger", "logs"),
#     "save-dir": os.path.join(directory, "dagger", "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall",
#     "expert-prob": 0.5,
# }
# counter = 0
# for learning_rate in [3e-3, 1e-3, 3e-4]:
#     for minibatch_size in [5120]:
#         for maxaggr_size in [102400, 51200]:
#             for numgames in [110, 5]:
#                 for gamma in [.995, 1.]:
#                     for seed in [1, 2]:
#                         j = {k:v for k,v in job.items()}
#                         subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                         run_name = "dagfx%d-%d" % (numgames, counter)
#                         j["state-directory"] = os.path.join(
#                             directory,
#                             "pomplays",
#                             subdir,
#                             "train")
#                         j["run-name"] = run_name
#                         j["seed"] = seed
#                         j["gamma"] = gamma
#                         j["minibatch-size"] = minibatch_size
#                         j["max-aggregate-agent-states"] = maxaggr_size
#                         j["seed"] = seed
#                         j["state-directory-distribution"] = "genesis"
#                         j["lr"] = learning_rate
#                         train_dagger_job(j, j["run-name"], is_fb=True, partition="uninterrupted")
#                         counter += 1


### Dagger again. 
# job = {
#     "num-processes": 60, "how-train": "dagger", "num-episodes-dagger": 10, "log-interval": 75,
#     "save-interval": 75, "num-steps-eval": 100,
#     "log-dir": os.path.join(directory, "dagger", "logs"),
#     "save-dir": os.path.join(directory, "dagger", "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall",
#     "expert-prob": 0.5,
# }
# counter = 0
# for learning_rate in [3e-3, 1e-3]:
#     for minibatch_size in [5120]:
#         for maxaggr_size in [51200]:
#             for numgames in [110, 5]:
#                 for gamma in [.995, 1.]:
#                     for seed in [3, 4, 5, 6]:
#                         for item_reward in [0., 0.1]:
#                             j = {k:v for k,v in job.items()}
#                             subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                             run_name = "dag2fx%d-%d" % (numgames, counter)
#                             j["state-directory"] = os.path.join(
#                                 directory,
#                                 "pomplays",
#                                 subdir,
#                                 "train")
#                             j["run-name"] = run_name
#                             if item_reward:
#                                 j["item-reward"] = item_reward
#                             j["seed"] = seed
#                             j["gamma"] = gamma
#                             j["minibatch-size"] = minibatch_size
#                             j["max-aggregate-agent-states"] = maxaggr_size
#                             j["seed"] = seed
#                             j["state-directory-distribution"] = "genesis"
#                             j["lr"] = learning_rate
#                             train_dagger_job(j, j["run-name"], is_fb=True, partition="uninterrupted")
#                             counter += 1


### Backselfplay First runs.
# These didn't work. Some hypothesis:
# 1. The agents were dumb at the beginning and so they just learned useless policies that didn't translate
#    when they then were introduced to complex agents later on.
# 2. The value loss is way off because it's an unstable and changing policy.
# job = {
#     "how-train": "backselfplay",  "log-interval": 7500, "save-interval": 100,
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4, 1e-4]:
#     for (name, distro) in [
#             ("uBnG", "uniformBoundsG"), #50
#             ("uBnJ", "uniformBoundsJ"), #75
#             ("uniform", "uniform"),
#     ]:
#         for numgames in [5]:
#             for itemreward in [0, .1]:
#                 for seed in [1, 2]:
#                     runng = 4
#                     j = {k:v for k,v in job.items()}
#                     subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                     log_dir = os.path.join(directory, "logs-fx%d-ubp" % runng)
#                     save_dir = os.path.join(directory, "models-fx%d-ubp" % runng)
#                     run_name = "bspubp-fx%d-%s-%d" % (runng, name, counter)

#                     j["use-both-places"] = ""
#                     j["state-directory"] = os.path.join(
#                         directory,
#                         "pomplays",
#                         subdir,
#                         "train")
#                     j["log-dir"] = log_dir
#                     j["save-dir"] = save_dir
#                     j["run-name"] = run_name
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1


### The above but with already trained policies. Note that we are using a policy that was trained only for the single agent. Will it do ok on the 2nd agent??? Unclear.
### These were pretty fucking dumb because we didn't restart the counts ...
### Start them over with that.
# job = {
#     "how-train": "backselfplay",  "log-interval": 7500, "save-interval": 100,
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4, 1e-4]:
#     for (name, distro) in [
#             ("uBnG", "uniformBoundsG"), #50
#             ("uBnJ", "uniformBoundsJ"), #75
#             ("uniform", "uniform"),
#             ("genesis", "genesis"),            
#     ]:
#         for numgames in [5]:
#             for itemreward in [0, .1]:
#                 for seed in [3]:
#                     runng = 4
#                     j = {k:v for k,v in job.items()}
#                     subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                     log_dir = os.path.join(directory, "logs-fx%d-ubp" % runng)
#                     save_dir = os.path.join(directory, "models-fx%d-ubp" % runng)
#                     run_name = "bspubpld-fx%d-%s-%d" % (runng, name, counter)

#                     j["use-both-places"] = ""
#                     j["state-directory"] = os.path.join(
#                         directory,
#                         "pomplays",
#                         subdir,
#                         "train")
#                     j["log-dir"] = log_dir
#                     j["save-dir"] = save_dir
#                     j["run-name"] = run_name
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                         j["restart-counts"] = ""
#                         j["saved-paths"] = os.path.join(directory, "models-fx4", "agent0-2fx4-uBnG-3.simple.FFACmp.Small.nc256.lr0.0003.bs102400.ns1707.gam1.0.seed3.gae.uniformBoundsG.itemrew0.100.epoch500.steps51210000.pt")
#                     else:
#                         j["restart-counts"] = ""                        
#                         j["saved-paths"] = os.path.join(directory, "models-fx4", "agent0-2fx4-uBnG-0.simple.FFACmp.Small.nc256.lr0.0003.bs102400.ns1707.gam1.0.seed3.gae.uniformBoundsG.epoch500.steps51210000.pt")
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
                    

### Here, we use the adaptive policy for frobackselfplay. We haven't done that in a long time so we try out a few differenta pproaches.
# job = {
#     "how-train": "frobackselfplay",  "log-interval": 7500, "save-interval": 100,
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000, "use-both-places": "",
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             ("uBnAdptA", "uniformBndAdptA"),
#             ("uBnAdptB", "uniformBndAdptB"),            
#             ("uniform", "uniform"),
#             ("genesis", "genesis"),            
#     ]:
#         for itemreward in [0, .1]:
#             for seed in [1, 2]:
#                 for use_saved_path in [False, True]:
#                     numgames = 5
#                     runng = 4
#                     j = {k:v for k,v in job.items()}
#                     subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                     log_dir = os.path.join(directory, "logs-fx%d-ubnadpt" % runng)
#                     save_dir = os.path.join(directory, "models-fx%d-ubnadpt" % runng)
#                     run_name = "fx%d-%s-%d" % (runng, name, counter)

#                     j["use-both-places"] = ""
#                     j["state-directory"] = os.path.join(
#                         directory,
#                         "pomplays",
#                         subdir,
#                         "train")
#                     j["log-dir"] = log_dir
#                     j["save-dir"] = save_dir
#                     j["run-name"] = run_name
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                         if use_saved_path:
#                             j["restart-counts"] = ""
#                             j["saved-paths"] = os.path.join(directory, "models-fx4", "agent0-2fx4-uBnG-3.simple.FFACmp.Small.nc256.lr0.0003.bs102400.ns1707.gam1.0.seed3.gae.uniformBoundsG.itemrew0.100.epoch500.steps51210000.pt")
#                     elif use_saved_path:
#                         j["restart-counts"] = ""                        
#                         j["saved-paths"] = os.path.join(directory, "models-fx4", "agent0-2fx4-uBnG-0.simple.FFACmp.Small.nc256.lr0.0003.bs102400.ns1707.gam1.0.seed3.gae.uniformBoundsG.epoch500.steps51210000.pt")
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
                    
                    
### Backplay but with adaptive simple training.
# job = {
#     "how-train": "simple",  "log-interval": 7500, "save-interval": 25,
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             ("uBnAdptA", "uniformBndAdptA"),
#             ("uBnAdptB", "uniformBndAdptB"),            
#     ]:
#         for numgames in [5]:
#             for itemreward in [0, .1]:
#                 for seed in [1, 2]:
#                     for use_second_place in [True, False]:
#                         runng = 4
#                         j = {k:v for k,v in job.items()}
#                         subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                         log_dir = os.path.join(directory, "logs-fx%d" % runng)
#                         save_dir = os.path.join(directory, "models-fx%d-adpt" % runng)
#                         run_name = "%s-%d" % (name, counter)
                        
#                         if use_second_place:
#                             j["use-second-place"] = ""
#                             subdir += "-2nd"
#                             log_dir += "usp"
#                             save_dir += "usp"
#                             run_name += "usp"
                            
#                         j["state-directory"] = os.path.join(
#                             directory,
#                             "pomplays",
#                             subdir,
#                             "train")
#                         j["log-dir"] = log_dir
#                         j["save-dir"] = save_dir
#                         j["run-name"] = run_name
#                         if itemreward:
#                             j["item-reward"] = itemreward
#                         j["seed"] = seed
#                         j["state-directory-distribution"] = distro
#                         j["lr"] = learning_rate
#                         train_ppo_job(j, j["run-name"], is_fb=True)
#                         counter += 1


### Like the above re adaptive but we are:
### 1. Busting out the per-agent success rates to be over only those games and not all games.
### 2. Restricting each thread to be roughyl onyl a single game. Two games are overrepresented.
### 3. Also doing mixed frozen+complex half the time.
# job = {
#     "how-train": "frobackselfplay",  "log-interval": 7500, "save-interval": 100,
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 64, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000, "use-both-places": "",
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             ("uBnAdptA", "uniformBndAdptA"),
#             ("uBnAdptB", "uniformBndAdptB"),    
#             ("uniform", "uniform"),
#             ("genesis", "genesis"),            
#     ]:
#         for itemreward in [0, .1]:
#             for seed in [1]:
#                 for use_saved_path in [False, True]:
#                     for mix_frozen_complex in [False, True]:
#                         numgames = 5
#                         runng = 4
#                         j = {k:v for k,v in job.items()}
#                         subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                         log_dir = os.path.join(directory, "logs-fx%d-ubnadpt" % runng)
#                         save_dir = os.path.join(directory, "models-fx%d-ubnadpt" % runng)
#                         run_name = "2fx%d-%s-%d" % (runng, name, counter)

#                         if mix_frozen_complex:
#                             j["mix-frozen-complex"] = ""
                            
#                         j["state-directory"] = os.path.join(
#                             directory,
#                             "pomplays",
#                             subdir,
#                             "train")
#                         j["log-dir"] = log_dir
#                         j["save-dir"] = save_dir
#                         j["run-name"] = run_name
#                         if itemreward:
#                             j["item-reward"] = itemreward
#                             if use_saved_path:
#                                 j["restart-counts"] = ""
#                                 j["saved-paths"] = os.path.join(directory, "models-fx4", "agent0-2fx4-uBnG-3.simple.FFACmp.Small.nc256.lr0.0003.bs102400.ns1707.gam1.0.seed3.gae.uniformBoundsG.itemrew0.100.epoch500.steps51210000.pt")
#                         elif use_saved_path:
#                             j["restart-counts"] = ""                        
#                             j["saved-paths"] = os.path.join(directory, "models-fx4", "agent0-2fx4-uBnG-0.simple.FFACmp.Small.nc256.lr0.0003.bs102400.ns1707.gam1.0.seed3.gae.uniformBoundsG.epoch500.steps51210000.pt")
#                         j["seed"] = seed
#                         j["state-directory-distribution"] = distro
#                         j["lr"] = learning_rate
#                         train_ppo_job(j, j["run-name"], is_fb=True)
#                         counter += 1


### Backplay like above with adaptive simple training, but here we lower the threshold to 0.5 instead of 0.6.
# job = {
#     "how-train": "simple",  "log-interval": 7500, "save-interval": 25,
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000, 'adapt-threshold': .5
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for itemreward in [0, .1]:
#         for numgames in [5]:
#             for (name, distro) in [
#                     ("uBnAdptA", "uniformBndAdptA"),
#             ]:
#                 for seed in [1, 2]:
#                     for use_second_place in [True, False]:
#                         for adapt_threshold in [.5, .6]:
#                             runng = 4
#                             j = {k:v for k,v in job.items()}
#                             j["adapt-threshold"] = adapt_threshold
#                             subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                             log_dir = os.path.join(directory, "logs-fx%d" % runng)
#                             save_dir = os.path.join(directory, "models-fx%d-adpt" % runng)
#                             run_name = "%s-%d" % (name, counter)
                            
#                             if use_second_place:
#                                 j["use-second-place"] = ""
#                                 subdir += "-2nd"
#                                 log_dir += "usp"
#                                 save_dir += "usp"
#                                 run_name += "usp"
                            
#                             j["state-directory"] = os.path.join(
#                                 directory,
#                                 "pomplays",
#                                 subdir,
#                                 "train")
#                             j["log-dir"] = log_dir
#                             j["save-dir"] = save_dir
#                             j["run-name"] = run_name
#                             if itemreward:
#                                 j["item-reward"] = itemreward
#                             j["seed"] = seed
#                             j["state-directory-distribution"] = distro
#                             j["lr"] = learning_rate
#                             train_ppo_job(j, j["run-name"], is_fb=True)
#                             counter += 1


### Running everything for 100 over for 5 seeds across uniform, genesis, and uBnL for the ICLR paper.
# job = {
#     "how-train": "simple",  "log-interval": 7500, "save-interval": 25,
#     "log-dir": os.path.join(directory, "logs-fxrp"), "save-dir": os.path.join(directory, "models"),
#     "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
#     "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
#     "num-frames": 2000000000,
# }
# counter = 0
# for learning_rate in [3e-4]:
#     for (name, distro) in [
#             ("uBnL", "uniformBoundsL"), #85
#             ("unfm", "uniform"), #85
#             ("gnss", "genesis")
#     ]:
#         for numgames in [110]:
#             for itemreward in [0, .1]:
#                 for seed in [1, 2, 3, 4, 5]:
#                     runng = 100
#                     j = {k:v for k,v in job.items()}
#                     subdir = "fx-ffacompetition%d-s100-complex" % numgames
#                     log_dir = os.path.join(directory, "logs-fx%d" % runng)
#                     save_dir = os.path.join(directory, "models-fx%d" % runng)
#                     run_name = "iclr%d-2fx%d-%s-%d" % (seed, runng, name, counter)
#                     j["state-directory"] = os.path.join(
#                         directory,
#                         "pomplays",
#                         subdir,
#                         "train")
#                     j["log-dir"] = log_dir
#                     j["save-dir"] = save_dir
#                     j["run-name"] = run_name
#                     if itemreward:
#                         j["item-reward"] = itemreward
#                     j["seed"] = seed
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True)
#                     counter += 1
                            

### Running everything for 4 over for 5 seeds across uniform, genesis, and uBnG for the ICLR paper.
### Additionally running use_second_place as well.
### There are 3*2*5*2 = 60 jobs here.
job = {
    "how-train": "simple",  "log-interval": 7500, "save-interval": 25,
    "log-dir": os.path.join(directory, "logs-fxrp"), "save-dir": os.path.join(directory, "models"),
    "config": "PommeFFACompetition-v0", "model-str": "PommeCNNPolicySmall", "use-gae": "",
    "num-processes": 60, "gamma": 1.0, "batch-size": 102400, "num-mini-batch": 20,
    "num-frames": 2000000000,
}
counter = 0
for learning_rate in [3e-4]:
    for (name, distro) in [
            ("uBnG", "uniformBoundsG"), #50
            ("unfm", "uniform"),
            ("gnss", "genesis")
    ]:
        for numgames in [5]:
            for itemreward in [0, .1]:
                for seed in [1, 2, 3, 4, 5]:
                    for use_second_place in [True, False]:
                        runng = 4
                        j = {k:v for k,v in job.items()}
                        subdir = "fx-ffacompetition%d-s100-complex-both" % numgames
                        log_dir = os.path.join(directory, "logs-fx%d" % runng)
                        save_dir = os.path.join(directory, "models-fx%d" % runng)
                        run_name = "iclr%d-2fx%d-%s-%d" % (seed, runng, name, counter)
                        
                        if use_second_place:
                            j["use-second-place"] = ""
                            log_dir += "usp"
                            save_dir += "usp"
                            run_name += "usp"
                            
                        j["state-directory"] = os.path.join(
                            directory,
                            "pomplays",
                            subdir,
                            "train")
                        j["log-dir"] = log_dir
                        j["save-dir"] = save_dir
                        j["run-name"] = run_name
                        if itemreward:
                            j["item-reward"] = itemreward
                        j["seed"] = seed
                        j["state-directory-distribution"] = distro
                        j["lr"] = learning_rate
                        train_ppo_job(j, j["run-name"], is_fb=True)
                        counter += 1
                    
