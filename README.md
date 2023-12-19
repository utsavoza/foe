<div align="center">
<h2>Adversarial Transferability Against Medical Deep Learning Systems</h2>
</div>

The robustness of medical deep learning systems against adversarial attacks is a critical
concern in the deployment of these systems in real-world clinical settings. Our project seeks
to understand the phenomenon of adversarial example transferability across different medical
imaging systems. We hypothesize that vulnerabilities to adversarial attacks may not be
domain-specific but could instead exhibit transferable properties that pose a broader risk to
medical deep learning systems.

## Usage

1. Clone the repository

   ```shell
   git clone git@github.com:utsavoza/foe.git
   cd foe
   ```

2. Download the validation and test datasets from [here](https://www.kaggle.com/datasets/nih-chest-xrays/data) and place
   them under `data/` directory (See the data section
   for more details).

3. Create and activate the virtual environment

   ```shell
   python -m venv venv
   source venv/bin/activate
   ```

4. Install the required dependencies

   ```shell
   pip install -r requirements.txt
   ```

5. Execute `craft_attacks.py` to reproduce the results, generate plots, etc.

   ```shell
   python craft_attacks.py --path_model='models/resnet50_finetuned_weights_final.hdf5' --save_results
   ```

## Data

1. Download the dataset from [here](https://www.kaggle.com/datasets/nih-chest-xrays/data). Be aware that the dataset is around 45GB in size, so ensure you have enough available storage space on your disk.


2. Extract the contents and arrange the dataset in the following structure under the `data/` directory. The `preprocess.py` script helps generate the data in the following format.

   ```
   data
   ├── images
   │   ├── train
   │   │   ├── No_Finding
   │   │   └── Pneumothorax
   │   └── val
   │       ├── No_Finding
   │       └── Pneumothorax
   └── transfer
       └── images
           ├── train
           │   ├── No_Finding
           │   └── Cardiomegaly
           └── val
               ├── No_Finding
               └── Cardiomegaly
   ```

## Crafting Attacks

- We explored the transferability of adversarial attacks in medical imaging using Projected Gradient Descent (PGD). We applied this technique to cardiomegaly detection in chest X-rays.

- Our strategy was to adapt adversarial modifications designed for pneumothorax detection to see if they could also mislead our AI in diagnosing cardiomegaly. This helped us understand the adaptability of such attacks across different medical conditions.

- The study revealed that our model is vulnerable to these cross-condition attacks. This finding is crucial for developing stronger defenses, ensuring more reliable medical diagnostics.

- To see how we did this, you can run the `craft_attacks.py` script. It will guide you through generating results and visualizations, demonstrating the model's response to these adversarial challenges.

## Repository Layout

| File/Directory       | Description                                                           |
|----------------------|-----------------------------------------------------------------------|
| `data`               | Datasets used for model training and testing                          |
| `images`             | Images, plots generated during the execution runs                     |
| `logs`               | Execution log files                                                   |
| `models`             | Saved finetuned models, weights, and checkpoint files                 |
| `notebooks`          | Jupyter notebooks for experimentation, data analysis, and prototyping |
| `scripts`            | Scripts to finetune model, and craft attacks on HPC                   |
| `craft_attacks.py`   | Script for generating adversarial attacks on models                   |
| `mixup_generator.py` | Tool for creating mixed data samples for data augmentation            |
| `preprocess.py`      | Script for preprocessing the data before training or testing          |
| `train.py`           | Main training script for finetuning resnet50                          |

## References

- [Adversarial Attacks Against Medical Deep Learning Systems](https://arxiv.org/pdf/1804.05296.pdf)

## Authors

- Jash Rathod (jsr10000)
- Utsav Oza (ugo1)

## License

This project is licensed under MIT License. See [LICENSE](./LICENSE).