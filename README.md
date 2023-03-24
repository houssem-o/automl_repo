### Pretrained deep ensembles for NAS

#### Installation
1. Clone this repo
2. Install dependencies. Python 3.9 was used, but the code should work with other versions.
`python -m pip install -r requirements.txt`
3. To generate the files used to pretrain and train the ensembles, other dependencies are needed. This is **optional**, as the data used is included in the `pretraining_data` directories. The optional installation is detailed in the last section below.

#### Logs and reported results
The log files track the evolution of quantities such as the best observed value and the rank correlations as the search procedure advances. Each run is logged in one pickled file, containing a list of dictionaries, each one representing the search state.
The folder `paper_logs_and_plots` contains the search logs and plots reported in the paper.

For easy viewing, the Jupyter notebook `logs.ipynb` explores these logs and displays the results and plots, including best values observed at (100, 200 and 400 evaluations), the evolution of the correlations, and statistics about when the optimum was reached.

The reported logs were obtained by running the experiments on an Apple M1 Pro CPU. The results are averaged over 10 runs.

**We include these logs as testing on Google Colab showed that running the full experiments took a long time, either on GPU or CPU. We are currently working on optimizations and improvements, but in our testing the pretraining procedure would take more than 3 hours on a Google Colab GPU, versus 30 minutes on the M1 Pro. As this might be too long in practice, we include these logs for the full procedure (1000 pretraining epochs), and provide code to run smaller scale experiments, which we detail in the following section.**

#### Code
Here, we provide the commands to run the experiments.
Run `cd automl_repo` once before these commands.
##### MLP experiments
`python -m experiments.nasbench_201.run_search
       [--experiment-name EXPERIMENT_NAME]
       [--runs RUNS]
       [--pretraining PRETRAINING]
       [--accelerator ACCELERATOR]
       [--threads THREADS]
       [--total-evals TOTAL_EVALS]
       [--pretrain-epochs PRETRAIN_EPOCHS]
       [--datasets DATASETS [DATASETS ...]]`
          
e.g. for a small-scale experiment, which takes about 40 minutes on a Tesla T4 (3 runs, on CIFAR10 only, with 20 pretraining epochs, with a maximum of 512 evaluations) run the following:
`python -m experiments.nasbench_201.run_search --accelerator gpu --datasets cifar10 --pretraining True --runs 3 --pretrain-epochs 20 --total-evals 512`

The full experiment, which takes about 35 minutes per dataset per run on an M1 Pro CPU, is the following (10 runs, datasets are CIFAR10 & CIFAR100 & ImageNet16-120, 1000 pretraining epochs, 1024 evaluations), is the following:
`python -m experiments.nasbench_201.run_search --runs 10 --pretrain-epochs 1000`

On the M1 Pro, we set the accelerator to CPU:
`python -m experiments.nasbench_201.run_search --runs 10 --pretrain-epochs 1000 --accelerator cpu`

A summary will be diplayed at the end of the experiment, and the logs will be saved in the `logs` folder.

##### GCN experiments
X

The same applies to the GCN experiments. For a small-scale experiment, which takes about **(((((((??)))))))** (3 runs, CIFAR10 only, 50 pretraining epochs, and a maximum of 512 evaluations), run the following:
`python -m experiments.gcn.run_search --accelerator gpu --datasets cifar10 --pretraining True --runs 3 --pretrain-epochs 50 --total-evals 512`

The full experiment takes about 90 minutes per run per dataset on the M1 Pro. The following command runs it (10 runs, CIFAR10 & CIFAR100 datasets, 1000 pretraining epochs, 1024 evaluations):
`python -m experiments.gcn.run_search --runs 10 --pretrain-epochs 1000 --accelerator cpu`

##### MO experiments
(CPU only for now)
`python -m experiments.mo.test_mo`

---

##### (Optional) Generating the pretraining and training data
The `generate_data.py` scripts contained in the folders of all three experiments generate the pickled files found in the `pretraining_data` directories. We chose to include the already generated files to avoid including the benchmarks in the requirements.txt file. However, these are their installation details.
###### MLP experiment files
The benchmark used is [NATS-Bench](https://github.com/D-X-Y/NATS-Bench "NATS-Bench"). Follow its installation instructions, and extract the [benchmark file](https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=sharing "Benchmark file") in the `experiments/nasbench_201/data` directory. (So the first file of the benchmark would be `experiments/nasbench_201/data/NATS-tss-v1_0-3ffb9-simple/000000.pickle.pbz2`)
Running `python -m experiments.nasbench_201.generate_data` generates the files which are then saved in the `pretraining_data` directory.
###### GCN experiment files
No files need to be generated for the GCN experiment, as it loads the MLP experiment files and converts them to a list of PyG_Data objects.
###### MO experiment files
The benchmark needed is [HW-NAS-Bench](https://github.com/GATECH-EIC/HW-NAS-Bench "HW-NAS-Bench"), installed by cloning the repository in the `experiments/hw_nas_bench` directory.