# SafeUnlearning ✨✅

Use SafeUnlearning to perform highly configurable, custom, and reproducible image-based machine unlearning tasks and evaluate unlearned model privacy risks, all from your Command Line Interface (CLI).

## Key Features
### Configurable Dataset Splitting
Download CIFAR datasets from Torchvision, or load image datasets in from your local drive. Define forget sets flexibly with a range of options from forgetting *n* samples, forgetting *k* classes, forgetting *n* samples from *k* classes, or forgetting specific filenames from your dataset.

### Model Training/Finetuning
Pretrained and non-pretrained `Torchvision` and `timm` models can be imported and trained, or finetuned to produce an original model before unlearning.

### Novel and Benchmark Unlearning and Privacy Attack Algorithms
SafeUnlearning includes a range of machine unlearning algorithms, reconstruction attacks and membership inference attacks which you can use to perform machine unlearning and privacy risk evaluation.

#### Unlearning Algorithms
* NegGrad, NegGrad+
* SCRUB
* Finetuning (Catastrophic Forgetting)
* Subspace Gradient Redirection Unlearning (SGRU)

#### Privacy Attack Algorithms
* Invert Gradient Reconstructor
* GGL
* GLiR
* LiRA

### Diverse Model Loading Options
SafeUnlearning supports model loading from Pytorch Hub, Torchvision, Pytorch Image Models and custom Pytorch modules.

### Powerful, Flexible Experiment Logging
After logging in to WandB through your terminal, SafeUnlearning allows you to optionally save training results to the WandB platform, dramatically improving your training logging and visualization capabilities.




## Who this project is for
This project is intended for two groups of users:
* Machine unlearning researchers who need a flexible pipeline containing SOTA algorithms to accelerate their research on new attacks and unlearning algorithms.

* Machine learning practitioners who need to perform machine unlearning jobs on custom models and datasets.



## Project dependencies
Before using this app, you will need to ensure you have the following prerequisites:
* Python 3.12.3


## Instructions for using SafeUnlearning
SafeUnlearning is structured as 4 mini-applications in 4 distinct folders: dataset splitting, model training, model unlearning, and reconstruction attacks/MIA on an unlearned model.

Each of the mini-applications required by your machine unlearning job can be run by calling the corresponding `main.py` file and specifying all required and any optional defaults from the mini-app's Hydra configuration setup. Hydra's documentation can be found [here](https://hydra.cc/docs/intro/).

Depending on your use-case, you are unlikely to need to use all of the mini-applications. For instance, if you are not interested in privacy risk and only need to perform unlearning on a pretrained model from Pytorch Hub, you only require the dataset splitting and unlearning mini-applications.

Walkthrough demonstrations of SafeUnlearning applied to different machine unlearning jobs can be found HERE.

### Example 1: Performing machine unlearning on pretrained MobileNetV2 on CIFAR10



### Example 2: Evaluating ResNet18's privacy risk on a custom dataset 

## Installation
1. Cloning this repository

    To clone this repository, run `git clone` from your CLI.

2. Python dependencies installation
 
    Create a virtual environment and install dependencies:

    ```bash
    python3 -m venv .venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
     ```

3. (Optional) WandB setup

    Generate a WandB API key and login to your WandB account from your CLI, so that your experiment logs can be logged and viewed on the WandB platform. Check out WandB's setup tutorial [here](https://docs.wandb.ai/quickstart/).


### Configure SafeUnlearning
SafeUnlearning is run using the Hydra framework. Hydra generates a single configuration from separate YAML files corresponding to different aspects of the job (e.g. the dataset, the model, the reconstruction attack algorithm).

You can override defaults and specify necessary commands from the CLI. Before trying to run a mini-app's `main.py`, we recommend you glance through their corresponding `src/<mini_app>/config` folder. The `config` folder should never be directly edited. Keys marked `???` are required arguments. Keys with an empty value are considered optional, and the corresponding key will not be used in the experiment if the value is left empty. For example, if `model_ckpt_path: ` is left empty, the app will not attempt to load existing model weights from the local drive. 


Keys with a non-empty value have that non-empty value as a default. A default or optionally empty value can be overridden from the CLI with `path.to.key=<new_value>`.

WandB is an optional but powerful add-on. To enable WandB logging during your experiment run, add on `+wandb=default` to your CLI command. 

Note: WandB is only available for the model training, model unlearning and privacy attack mini-apps.

### Run SafeUnlearning
Once you know your experiment parameters, you can then run the appropriate mini-app's `main.py` file to perform that sub-task of your overall unlearning job. 

For instance, suppose your experiment involves forgetting 10 samples from class 0 of CIFAR10, and using 20% of the data as validation. You would run the following command to download the CIFAR10 binaries to the `./raw` folder and your image dataset's train, validation, test, forget and retain splits to the `./data` folder:
```bash
python src/datasets/main.py forget=classnum dataset=cifar10 dataset.load_method=torchvision dataset.binaries_download_dir=./artifacts forget.forget_idx="{0:10}" dataset.init_path=./raw dataset.save_path=./data experiment_name=test
```

## Contributing guidelines
The contributing guidelines for this project are TBD, as the software license has not yet been determined.


## Troubleshooting
### You must specify `abc`, e.g. `abc=<OPTION>`
This is a common but easily understandable error thrown by Hydra when you run `python src/mini_app/main.py` without specifying a sub-config file for an important aspect of the overall configuration. For instance, running exactly `python src/unlearning/main.py` returns:

```bash
You must specify 'unlearner', e.g, unlearner=<OPTION>
Available options:
        cfk
        euk
        finetune
        neggrad
        neggradplus
        scrub
        sgru

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

To solve this issue, you should specify `python src/unlearning/main.py unlearner=<unlearner_name>`. Typically, further sub-configs may be missing, causing the same error to re-appear for the next missing sub-config until you have specified all required sub-configs (e.g. `dataset=<option>`, `model=<option>`). If you still have trouble understanding how to solve this issue, we would strongly recommend you to walk through the Hydra tutorial [here](https://hydra.cc/docs/intro/).

### `MissingMandatoryValue` error when running `main.py`.
This is a common error you will encounter when you have not provided a necessary default value for your experiment configurations. For example, running 
```bash
python src/unlearning/main.py dataset=cifar10 model=torchhub unlearner=neggradplus dataset.save_path=./data output_dir=./artifacts/models +wandb=default
```
will yield the following error:
```bash
Error executing job with overrides: ['dataset=cifar10', 'model=torchhub', 'unlearner=neggradplus', 'dataset.save_path=./data', 'output_dir=./artifacts/models', '+wandb=default']
Traceback (most recent call last):
  File ".../src/unlearning/main.py", line 212, in main
    raise MissingMandatoryValue(
omegaconf.errors.MissingMandatoryValue: Missing the following required arguments in the configuration: {'model.model_name', 'model.repo_path'}. 
Hint: python file.py key=value sets the appropriate value.
```
The error indicates that we have specified that we want to use a pretrained model loaded from Pytorch Hub, but failed to specify the model name and repo. To solve this error, just follow the hints and add on the missing required arguments `model.model_name=<MODELNAME>` and `model.repo_path=<REPOPATH>` in the shell command. As a useful addition, SafeUnlearning will also have printed the run configuration into the terminal. As you can see, the `model.model_name` and `model.repo_path` keys are required `???` but not given a value, which will raise a `MissingMandatoryValue` exception.

```bash
============ Run Configuration ============
seed: 42
verbose: true
output_dir: ./artifacts/models
id: null
model:
  load_method: torchhub
  repo_path: ???
  model_name: ???
  model_ckpt_path: null
dataset:
  name: cifar10
  num_classes: 10
  save_path: ./data
  cfg:
    batch_sizes:
      retain: 128
      forget: 128
      val: 128
    num_workers: 4
unlearner:
  name: neggradplus
  evaluate: true
  cfg:
    optimizer: sgd
    lr: 0.001
    momentum: null
    lr_decay_factor: null
    epochs_per_lr_decay: null
    weight_decay: 0.0005
    use_l2_penalty: false
    epochs: 5
    beta: 0.99
wandb:
  project_name: machine_unlearning_001
  run_id: null

============================================
```

## Authors
* Andrew Siow (andrew.siow24@imperial.ac.uk)
* Bobbi Halsell (bobbi.halsell24@imperial.ac.uk)
* Varsha Bubathi (varsha.bubathi24@imperial.ac.uk)
* Ola Pasieka (aleksandra.pasieka24@imperial.ac.uk)
* Jebastin Nadar (jebastin.nadar24@imperial.ac.uk)
* Lynn Liu (yixuan.liu118@imperial.ac.uk)

This software was developed as part of a Masters' group project at Imperial College London in 2025, under the supervision of Mohammed Maheri (m.maheri23@imperial.ac.uk) and Professor Hamed Haddadi (h.haddadi@imperial.ac.uk). 

## Terms of use
The SafeUnlearning app license is TBD.

---