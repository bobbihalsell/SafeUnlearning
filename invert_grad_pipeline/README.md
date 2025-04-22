# Invert Gradient Pipeline

This directory contains the pipeline for running experiments related to safe unlearning using the invert gradient method. Below are the steps to execute the pipeline and run the final experiments.
#TODO: what is inversegradient

what the pipeline is done on: Resnet18 trained from scratch on CIFAR-10 data. Up to 32 samples in forget folder, 50 retain.
In the experimetns you can specify the number of samples you want to unlearn (up to 32) and this runs configures the data folders, unlearns the models and save results on WandB. 

results:
## Steps to Run the Experimentation Pipeline

1. **Generate Settings**  
    Run the following command to generate the necessary settings for the pipeline:  
    ```bash
    bash invert_grad_pipeline/step_0_generate_settings.sh
    ```

2. **Generate Forget Indices**  
    Use the following command to generate the indices of data to forget:  
    ```bash
    python invert_grad_pipeline/step_1_generate_forget_idx.py
    ```

3. **Data Splitting**  
    Split the data into training, validation, and test sets by running:  
    ```bash
    bash invert_grad_pipeline/step_2_data_splitting.sh
    ```

4. **Training**  
    Train the model using the following command:  
    ```bash
    bash invert_grad_pipeline/step_3_training.sh
    ```

5. **Create Pool**  
    Create the pool of data for experiments by executing:  
    ```bash
    bash invert_grad_pipeline/step_4_create_pool.sh
    ```

## Running Final Experiments

To run the final experiments, use the following commands:

1. For single sample NegGrad+ experiments:  
    ```bash
    python invert_grad_pipeline/exp_main.py -m samples=1 epochs=1,2 method=neggradplus seed=0 beta=0.1,0.2,0.3,0.4 retain=5,15
    ```

2. For experiments with 16 samples on NegGrad+:  
    ```bash
    python invert_grad_pipeline/exp_main.py -m samples=16 epochs=1,2 method=neggradplus seed=0 beta=0.1,0.2,0.3,0.4 retain=25,50
    ```

Follow the above steps sequentially to ensure the pipeline runs correctly.  