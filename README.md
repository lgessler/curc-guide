(This guide is intended for my [CU SPUR](https://www.colorado.edu/engineering/students/research-opportunities/summer-program-undergraduate-research-cu-spur) collaborators in summer of 2024!)

# 1. Introduction
Modern NLP systems typically employ the methods of deep learning. 
This has the practical consequence that they are so computationally intensive that they need a graphics processing unit (GPU, aka "graphics card") in order to be run.
We use CU Research Computing (CURC) in order to run our jobs that require GPUs.

CURC is a division of the university that maintains large fleets of computers that have GPUs and other high-performance computing devices. 
As CU affiliates, we can use these for our research for free, though it requires some additional setup.

Here, I'll show you how to get set up on CURC and walk you through running a job.
You can regard this document as a distillation of [the official CURC docs](https://curc.readthedocs.io/en/latest/index.html) with some additional content added.
If you're wondering about anything, you can either check there or ask me.

# 2. CURC and Alpine Overview
Follow [the official docs' guide](https://curc.readthedocs.io/en/latest/access/logging-in.html) to getting set up and logged in.

## File System
There is a [shared file system](https://curc.readthedocs.io/en/latest/compute/filesystems.html) accessible from any CURC cluster and node.
You will care about three locations:

1. `/home/$USER`: this is limited to 2GB and should be used only to store things like application configurations, credentials, binaries, and other small files.
2. `/projects/$USER`: this is limited to 250GB and is where things like your Python environment, datasets, and code will be stored.
3. `/scratch/alpine/$USER`: this is limited to 10TB and is intended for **temporary** files that your program might produce during execution. You will not need this most of the time.

All these locations are accessible in the usual way via the command line after you've logged in.
But note that you can also use [OnDemand](https://ondemand.rc.colorado.edu/) to access your files in a convenient web GUI.

## Alpine
[Alpine](https://curc.readthedocs.io/en/latest/clusters/alpine/index.html) is one of the three major clusters that CURC maintains, and it is the one we will be using.

### Node Types
Alpine has four [node types](https://curc.readthedocs.io/en/latest/compute/node-types.html) which are specialized into different kinds of work. These are:

1. Login nodes: this is where you are after `ssh`ing into `login.rc.colorado.edu`. They are not used for any heavy computation.
2. Compile nodes: these were originally intended for compiling code for languages where that is an expensive operation, like C++ or Fortran. That doesn't apply to us, but for this historical reason, you still need to access a compile node in order to do things like e.g. set up new Anaconda environments.
3. Compute nodes: these are the ones with the GPUs that run your jobs. While you can connect to them directly, you will primarily be sending jobs to them instead.
4. Data transfer nodes: don't worry about these--we'll use some other means to move files around.

### Software

Software that requires non-trivial installation on the system is managed using [`module`](https://curc.readthedocs.io/en/latest/compute/modules.html).
By default, almost no packages will be made available to you--instead, you will dynamically load them a la carte using the `module` command.
For example, `module load anaconda` can be used to load Anaconda and gain access to the `conda` command.

Note that on Alpine, you must connect to a compile node via `acompile` before most modules will be available to you.

# 3. Tutorial: Fine-tuning RoBERTa

Let's get set up and run a real job on Alpine! 

(Note: I'll be using `vim` to edit files in the terminal, but feel free to use whatever terminal text editor you like. If you really hate using terminal text editors, you could also use [OnDemand](https://ondemand.rc.colorado.edu/) to edit files.)


## Connecting
First, log in to CURC and connect to a compile node:

```bash
ssh xxxxdddd@login.rc.colorado.edu
acompile
```

For convenience, I recommend creating a symlink between your `/home` and `/projects` directory so that `cd ~/projects` will take you to `/home/$USER/projects`, though this is not necessary:

```bash
ln -s /projects/$USER /home/$USER/projects
```

## Anaconda Configuration
Now we want to [set up Anaconda](https://curc.readthedocs.io/en/latest/software/python.html?highlight=anaconda). 
Before that, we need to do some setup to make sure that the large files they create go somewhere that's not `/home`.

For Anaconda, create the file `~/.condarc` and paste the following:

```bash
pkgs_dirs:
  - /projects/$USER/.conda_pkgs
envs_dirs:
  - /projects/$USER/software/anaconda/envs
```

For `pip`, execute the following code to add an entry to the end of your `~/.bashrc`:

```bash
echo 'alias pip="XDG_CACHE_HOME=/projects/$USER/software/pip pip"' >> .bashrc
```

Now, use `module` to load Anaconda:
```bash
module add anaconda
```
Verify that the load was successful by running `python`.

## Environment Setup

Now, let's make a new Python environment and add some dependencies:

```bash
conda create --name hf python=3.10
conda activate hf
pip install "transformers[torch]" datasets evaluate numpy scikit-learn
```

Let's put our code (a simplified version of [this tutorial code from HuggingFace](https://huggingface.co/docs/transformers/en/training)) under the projects folder:

```bash
mkdir /projects/$USER/hf_demo
cd /projects/$USER/hf_demo
wget https://raw.githubusercontent.com/lgessler/curc-guide/main/main.py
```

## Slurm Setup
Things are not as simple as running `python main.py` now. 
You are currently logged into a compile node, and compile nodes do not have GPUs.
Instead, you will submit a job to [Slurm](https://curc.readthedocs.io/en/latest/running-jobs/running-apps-with-jobs.html), a system that manages job submissions.

Slurm is a system that takes specifications of jobs and manages how those jobs are executed. The number of jobs typically far outstrips the number of GPUs that are available, so a central system needs to control the process of which jobs get released to which nodes at what times.

First, create two directories under `/projects/$USER`:

```bash
mkdir logs
mkdir scripts
```

Now, create a new file `scripts/hf_demo.sh` and fill it with the following:


```bash
#!/bin/bash
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=1:00:00
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --job-name=hf-demo
#SBATCH --output=logs/hf-demo-%j.output

module purge
module add anaconda

cd hf_demo
# Using `conda activate` inside scripts can be annoying--just use the absolute path to the `python` bin instead
/projects/$USER/software/anaconda/envs/hf/bin/python main.py
```

This is a script for submitting a [batch job](https://curc.readthedocs.io/en/latest/running-jobs/batch-jobs.html).
As you can see, the script looks like a normal `sh` script except for all the commented lines at the top, which are telling Slurm important information about this job. 
Briefly:

* `account`: all jobs are associated with an account (or an "allocation") which tracks the number of **service units** (SUs) that have been used for a particular project. Service units are "device-hours" which you can spend on CPUs and GPUs. Speaking loosely, the more SUs you use, the *lower priority* your jobs will be, meaning some others will be able to queue jobs before you during times of high demand. By default, you will use the `ucb-general` allocation.
* `nodes`: this is the number of devices needed to run the job. You will always have this set to 1.
* `ntasks`: this is the number of CPU cores assigned to your job, and this will also determine the amount of CPU RAM your job will have. Consider decreasing if your job won't need much CPU contribution.
* `time`: the maximum amount of time your job will run for. Your job will be killed after this time is exceeded. The maximum time that can be standardly assigned is 24h.
* `partition`: the [subcluster of Alpine](https://curc.readthedocs.io/en/latest/clusters/alpine/alpine-hardware.html) that will process your job. For GPU jobs, you will always keep this on `aa100`.
* `gres`: this is used to specify the number of GPUs you need. You will always keep this at one.
* `job-name`: a meaningful human-readable name for your job. Name it after the project you're running the job for.
* `output`: the path to a log file that will receive a pipe of the `stdout` produced by the job. Note that the `%j` in the name will be replaced by the numeric ID of the job, which will prevent collisions across jobs.

## Running with Slurm
Now, run `sbatch scripts/hf_demo.sh`.
This submits your job to Slurm, and it will now wait until a node is available to execute your job.
You can track your job's status by entering `squeue --user=$USER`. 
After it's done, you will no longer see it with `squeue`. 
Your job's output will be visible under `logs/` with the file name you specified.

If all goes well, you should now see output from your first Slurm job!

## Cheatsheet
CURC has a [cheatsheet](https://curc.readthedocs.io/en/stable/additional-resources/CURC-cheatsheet.html) which you may find helpful.
