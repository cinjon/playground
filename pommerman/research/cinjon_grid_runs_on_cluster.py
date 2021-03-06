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

### First djikstra runs.
# job = {
#     "how-train": "grid",  "log-interval": 5000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99, "batch-size": 25600, "num-mini-batch": 5,
#     "state-directory": os.path.join(directory, "astars110-s100/train"),
# }
# counter = 0
# for learning_rate in [1e-4, 3e-4, 1e-3]:
#     for (name, distro) in [
#             ("genesis", "genesis"),
#             ("uBnGrA", "uniformBoundsGrA"),
#             ("uBnGrB", "uniformBoundsGrB"),
#     ]:
#         for step_loss in [.03, .1]:
#             for seed in [1]:
#                 j = {k:v for k,v in job.items()}
#                 j["run-name"] = "grid-%s-%d" % (name, counter)
#                 j["seed"] = seed
#                 j["step-loss"] = step_loss
#                 j["state-directory-distribution"] = distro
#                 j["lr"] = learning_rate
#                 train_ppo_job(j, j["run-name"], is_fb=True, partition="learnfair")
#                 counter += 1


### Do the Djikstra runs that worked but with rigid=180, ml30 and online30.
# In particular --> grUniformBounds{A, B}, lr 1e-3, stl of .03
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for learning_rate in [3e-3, 1e-3]:
#     for (name, distro) in [
#             ("genesis", "genesis"),
#             ("grUBnA", "grUniformBoundsA"),
#             ("grUBnB", "grUniformBoundsB"),
#     ]:
#         for step_loss in [.01, .03]:
#             for seed in [1, 2]:
#                 for state_directory in [
#                         "online", os.path.join(directory, "ml30-astars110-s100/train")
#                 ]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     if state_directory == "online":
#                         j["log-dir"] += "-online"
#                         j["run-name"] = "onlMl30-%s" % j["run-name"]
#                         j["log-interval"] = 1000
#                     else:
#                         j["log-dir"] += "-100"
#                         j["run-name"] = "100Ml30-%s" % j["run-name"]                        
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["state-directory"] = state_directory
#                     j["lr"] = learning_rate
#                     time = 72 if state_directory == "online" else 36
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1


### Do the Djikstra runs that worked but with rigid=180, ml25 and online25.
# In particular --> grUniformBounds{A, B}, lr 1e-3, stl of .03
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for learning_rate in [1e-3]:
#     for (name, distro) in [
#             ("genesis", "genesis"),
#             ("grUBnA", "grUniformBoundsA"),
#             ("grUBnB", "grUniformBoundsB"),
#             ("grUBnB", "grUniformBoundsC"),            
#     ]:
#         for step_loss in [.03]:
#             for seed in [1, 2, 3]:
#                 for state_directory in [
#                         "online", os.path.join(directory, "ml25-astars110-s100/train")
#                 ]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     if state_directory == "online":
#                         j["log-dir"] += "-online"
#                         j["run-name"] = "onlMl25-%s" % j["run-name"]
#                         j["log-interval"] = 1000
#                     else:
#                         j["log-dir"] += "-100"
#                         j["run-name"] = "100Ml25-%s" % j["run-name"]                        
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["state-directory"] = state_directory
#                     j["lr"] = learning_rate
#                     time = 36 if state_directory == "online" else 24
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1
                    

### Do the Djikstra runs that worked but with rigid=120 and ml25.
# In particular --> grUniformBounds{A, B}, lr 1e-3, stl of .03
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for learning_rate in [1e-3]:
#     for (name, distro) in [
#             ("genesis", "genesis"),
#             ("grUBnA", "grUniformBoundsA"),
#             ("grUBnB", "grUniformBoundsB"),
#     ]:
#         for step_loss in [.03]:
#             for seed in [1, 2, 3]:
#                 for state_directory in [
#                         os.path.join(directory, "ml25-120-astars110-s100/train")
#                 ]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     if state_directory == "online":
#                         j["log-dir"] += "-online"
#                         j["run-name"] = "onlMl25-120-%s" % j["run-name"]
#                         j["log-interval"] = 1000
#                     else:
#                         j["log-dir"] += "-100"
#                         j["run-name"] = "100Ml25-120-%s" % j["run-name"]                        
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["state-directory"] = state_directory
#                     j["lr"] = learning_rate
#                     time = 36 if state_directory == "online" else 24
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1
                    
                    
### Do the Djikstra runs that worked but with rigid=120 and ml15.
# In particular --> grUniformBounds{A, B}, lr 1e-3, stl of .03
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for learning_rate in [1e-3]:
#     for (name, distro) in [
#             ("genesis", "genesis"),
#             ("grUBnA", "grUniformBoundsA"),
#             ("grUBnB", "grUniformBoundsB"),
#     ]:
#         for step_loss in [.03]:
#             for seed in [1, 2, 3]:
#                 for state_directory in [
#                         os.path.join(directory, "ml15-120-astars100-s100/train"),
#                         os.path.join(directory, "ml15-120-astars500-s100/train")
#                 ]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     if "astars100" in state_directory:
#                         j["log-dir"] += "-100"
#                         j["run-name"] = "100Ml15-120-%s" % j["run-name"]
#                     elif "astars500" in state_directory:
#                         j["log-dir"] += "-500"
#                         j["run-name"] = "500Ml15-120-%s" % j["run-name"]
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["state-directory"] = state_directory
#                     j["lr"] = learning_rate
#                     time = 36 if state_directory == "online" else 24
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1


### Redo the Djikstra runs from the folder that worked. This is rigid=120, but ignoring
# the min_path shit --> grUniformBounds{A, B}, lr 1e-3, stl of .03
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "state-directory": os.path.join(directory, "astars110-s100"),
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for state_directory in [
#         "",
#         "online"
# ]:
#     for (name, distro) in [
#             ("grUBnB", "grUniformBoundsB"),
#             ("genesis", "genesis"),
#     ]:
#         for seed in [3, 4, 5]:
#             for learning_rate in [1e-3]:
#                 for step_loss in [.03]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     j["run-name"] = "grid21-%s" % j["run-name"]
#                     if state_directory == "online":
#                         j["run-name"] += "-onl"
#                         j["log-interval"] = 1000
#                         j["log-dir"] += "-online"
#                         j["state-directory"] = state_directory
#                         time = 72
#                     else:
#                         j["run-name"] += state_directory
#                         j["state-directory"] += "%s/train" % state_directory
#                         time = 24
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1
                    

### Redo the Djikstra runs from the folder that worked. This is rigid=120, but ignoring
# the min_path shit --> grUniformBounds{A, B}, lr 1e-3, stl of .03.
# Do these with the 5opt and 10opt.
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "state-directory": os.path.join(directory, "astars110-s100"),
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for state_directory in [
#         # "-5opt",
#         "-10opt",
# ]:
#     for (name, distro) in [
#             ("grUBnB", "grUniformBoundsB"),
#             ("genesis", "genesis"),
#     ]:
#         for seed in [3, 4, 5]:
#             for learning_rate in [1e-3]:
#                 for step_loss in [.03]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     j["run-name"] = "grid21-%s" % j["run-name"]
#                     if state_directory == "online":
#                         j["run-name"] += "-onl"
#                         j["log-interval"] = 1000
#                         j["log-dir"] += "-online"
#                         j["state-directory"] = state_directory
#                         time = 72
#                     else:
#                         j["run-name"] += state_directory
#                         j["state-directory"] += "%s/train" % state_directory
#                         time = 24
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1


### Redo the Djikstra runs from the folder that worked. This is rigid=120, but ignoring
# the min_path shit --> grUniformBounds{A, B}, lr 1e-3, stl of .03.
# Do these with the 5opt and 10opt.
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "state-directory": os.path.join(directory, "astars110-s100"),
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for state_directory in [
#         "-5opt",
#         # "-10opt",
# ]:
#     for (name, distro) in [
#             ("grUBnB", "grUniformBoundsB"),
#             ("genesis", "genesis"),
#     ]:
#         for seed in [3, 4, 5]:
#             for learning_rate in [1e-3]:
#                 for step_loss in [.03]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     j["run-name"] = "grid21-%s" % j["run-name"]
#                     if state_directory == "online":
#                         j["run-name"] += "-onl"
#                         j["log-interval"] = 1000
#                         j["log-dir"] += "-online"
#                         j["state-directory"] = state_directory
#                         time = 72
#                     else:
#                         j["run-name"] += state_directory
#                         j["state-directory"] += "%s/train" % state_directory
#                         time = 24
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1
                    

# Run these with uniform step choices.
# job = {
#     "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
#     "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
#     "save-dir": os.path.join(directory, "models"), "num-channels": 32,
#     "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
#     "num-processes": 60, "gamma": 0.99,
#     "state-directory": os.path.join(directory, "astars110-s100"),
#     "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
# }
# counter = 0
# for state_directory in [
#         "",
#         "-5opt",
#         "-10opt",
# ]:
#     for (name, distro) in [
#             ("uniform", "uniform"),
#     ]:
#         for seed in [3, 4, 5]:
#             for learning_rate in [1e-3]:
#                 for step_loss in [.03]:
#                     j = {k:v for k,v in job.items()}
#                     j["run-name"] = "%s-%d" % (name, counter)
#                     j["run-name"] = "grid21-%s" % j["run-name"]
#                     j["run-name"] += state_directory
#                     if state_directory == "":
#                         j["state-directory"] = os.path.join(
#                             directory, "ml15-120-astars100-s100", "train")
#                         j["log-dir"] += "-100"
#                     else:
#                         j["state-directory"] += "%s/train" % state_directory                        
#                     time = 24
#                     j["seed"] = seed
#                     j["step-loss"] = step_loss
#                     j["state-directory-distribution"] = distro
#                     j["lr"] = learning_rate
#                     train_ppo_job(j, j["run-name"], is_fb=True,
#                                   partition="uninterrupted", time=time)
#                     counter += 1
                    

### Running everything seeds 1,2 for everything except 4,5 for reg-grUBnB for the ICLR paper.
### Reg is the dataset being the optimal one.
job = {
    "how-train": "grid",  "log-interval": 10000, "save-interval": 100,
    "log-dir": os.path.join(directory, "logs"), "num-stack": 1,
    "save-dir": os.path.join(directory, "models"), "num-channels": 32,
    "config": "GridWalls-v4", "model-str": "GridCNNPolicy", "use-gae": "",
    "num-processes": 60, "gamma": 0.99,
    "state-directory": os.path.join(directory, "astars110-s100"),
    "batch-size": 102400, "num-mini-batch": 20, "num-frames": 2000000000
}
counter = 0
for state_directory in [
        "",
        "-5opt",
        "-10opt",
]:
    for (name, distro) in [
            ("grUBnB", "grUniformBoundsB"),
            ("gnss", "genesis"),
            ("unfm", "uniform"),
    ]:
        for seed in [1, 2, 4, 5]:
            if state_directory == "" and name == "grUBnB":
                if seed < 4:
                    continue
            else:
                if seed > 2:
                    continue
                
            for learning_rate in [1e-3]:
                for step_loss in [.03]:
                    j = {k:v for k,v in job.items()}
                    j["run-name"] = "%s-%d" % (name, counter)
                    j["run-name"] = "iclr%d-grid21-%s" % (seed, j["run-name"])
                    j["run-name"] += state_directory
                    if state_directory == "":
                        j["state-directory"] = os.path.join(
                            directory, "ml15-120-astars100-s100", "train")
                        j["log-dir"] += "-100"
                    else:
                        j["state-directory"] += "%s/train" % state_directory                        
                    time = 24
                    j["seed"] = seed
                    j["step-loss"] = step_loss
                    j["state-directory-distribution"] = distro
                    j["lr"] = learning_rate
                    train_ppo_job(j, j["run-name"], is_fb=True,
                                  partition="uninterrupted", time=time)
                    counter += 1
                    
                    
