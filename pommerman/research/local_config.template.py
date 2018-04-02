cluster_directory = '/path/to/directory/on/cluster'
email = 'your-email@nyu.edu'

def write_extra_sbatch_commands(f):
    f.write("cd ${HOME}/Code/venvs" + "\n")
    f.write("source selfplayground/bin/activate" + "\n")
    f.write("SRCDIR=${HOME}/Code/selfplayground" + "\n")
    f.write("cd ${SRCDIR}/pommerman/research" + "\n")
