# Triager X

Triager X is a novel triaging framework designed to streamline the process of handling GitHub issues. By analyzing the title and description of a GitHub issue, and using two rankers (1) TriagerX Content-based Ranker (CBR), (2) TriagerX Interaction-based Ranker (IBR), Triager X recommends the most appropriate developers to address the issue. This tool can significantly reduce the time and effort needed to manage and assign issues in software projects.

## Features
- **Automatic Component Recommendation**: Suggests relevant components for each GitHub issue based on its content.
- **Developer Assignment**: Identifies and recommends developers best suited to handle the issue.
- **Efficient Issue Management**: Enhances productivity by automating the triaging process, allowing teams to focus on resolving issues rather than sorting them.

## Environmental Setup
Create a new `conda` environment by executing the following command:

```bash
conda env create -f environment.yml
```
Activate the environment by:

```bash
conda activate myenv
```

## Preparing the Datasets
Download the bug datasets compressed in this zip file: [**data.zip - Google Drive**](https://drive.google.com/file/d/1UUcPWk0nO31xlTyZnhy88hJ7yq3MmkZ-/view?usp=drive_link)

Once the compressed zip is downloaded, extract it to a suitable directory. The directory structure should be something similar to this:
```bash
data/
├── deeptriage/
│   ├── google_chrome/
│   ├── mozilla_core/
│   └── mozilla_firefox/
├── openj9/
│   ├── issue_data/
│   └── openj9_bug_data.csv
└── typescript/
    ├── issue_data/
    └── ts_bug_data.csv
```
The data/datasets inside `deeptriage` directory is from [literature](https://dl.acm.org/doi/abs/10.1145/3297001.3297023) and collected from [this repository](https://github.com/hacetin/deep-triage). This can only be used with the TriagerX CBR.

`openj9` and `typescript` directories contain the datasets we prepared. Each of these directories have complete github issues data with interaction events in `json` format. While to train or run TriagerX, no further data preparation is required as we are providing it, issue data can be downloaded using Github API from any repository using the following pipeline:

```bash
python util/issue_dump.py --path data/openj9 --owner openj9 --repo openj9
```
However, a github token is required. We set the token on `.env` with `GH_TOKEN` key.

Once the issue data is downloaded, we can use this data for TriagerX IBR component and prepare it for TriagerX CBR training. Similarly, `*_bug_data.csv` is used for training TriagerX CBR where we have issue summaries, descriptions and the bug fixers. This CSV datasets can be prepared by using `util/make_dataset.py` script. We just need to refer the JSON directory root (e.g., `data/openj9/issue_data`) and the output CSV path. This step is also not required if the provided data is used.


## Training
We provide seperate training pipelines for each dataset. All the training scripts can be found in `training/developer` directory.

Training scripts require training configs defined in `yaml` files. All of our training configs can be found under `training/training_config`.

Some configuration parameters are described below:

```yaml
use_description: true
# Boolean flag indicating whether to use descriptions within the model. When False, the model will be trained with only bug title/summary.

base_transformer_models: 
 - "microsoft/deberta-base"
 - "roberta-base"
# List of pre-trained transformer model keys from the HuggingFace repository to be used as base models for the ensemble. Multiple models to be used only when we are training TriagerX configuration.

unfrozen_layers: 3
# Number of layers of PLM to unfreeze during training. Unfrozen layers will be fine-tuned while the rest remain frozen.

num_classifiers: 3
# Number of classifiers to include in the ensemble. This parameter defines how many separate classifiers will be trained and used for predictions.

dropout: 0.2
# Dropout rate to be applied in the neural network layers to prevent overfitting. A value of `0.2` means a 20% dropout rate.

learning_rate: 0.00001
# Maximum learning rate to be used by AdamW optimizer.

max_tokens: 256
# Maximum number of tokens to be considered in each input sequence. This parameter limits the input length to `256` tokens for all PLMs.

topk_indices: 
  - 3
  - 5
  - 10
  - 20
# List of top-k indices to be evaluated. These values specify the positions (k) at which performance metrics will be calculated (e.g., top-3, top-5, top-10, top-20).

model_key: "triagerx"
# Key identifier for the model. This is a unique string used by ModuleFactory. Options are "triagerx," "cnn-transformer," and "fcn-transformer."

run_name: "triagerx_openj9"
# Name for the current training run. This helps to distinguish different runs and experiments, here named "triagerx_openj9".

weights_save_location: "~path/to/weights"
# File path where the trained model weights will be saved. The tilde (`~`) indicates the home directory.

test_report_location: "~path/to/evaluation_report"
# File path where the evaluation report will be saved after testing. The tilde (`~`) indicates the home directory.

wandb_project: "wandb_project_key"
# Project key for logging experiments to Weights & Biases (Wandb). This key links the run to a specific project on Wandb.
```

Once the training configuration is ready, training can simply be started by:
```bash
python training/developer/developer_training_openj9.py \
        --config training/training_config/openj9/triagerx.yaml \
        --dataset_path data/openj9_bug_data.csv \
        --seed 42
```
_In our experimental setup, we used SLURM to train the models. Those scripts can be found under `scripts` directory._

## TriagerX Recommendation Generation
Once the required CBR model is trained, recommendations can be generated using [`TriagerX`](triagerx/system/triagerx.py) pipeline.

To import the pipeline:
```python
from triagerx.system.triagerx import TriagerX

# Initialize the pipeline
triagerx_pipeline = TriagerX(
    developer_prediction_model=TRAINED_DEV_MODEL,
    similarity_model=PRETRAINED_SENTENCE_TRANSFORMER_MODEL,
    issues_path=PATH_TO_STORED_JSON_DATA_OF_GITHUB_ISSUES,
    train_embeddings=PATH_TO_SAVED_TRAIN_DATA_EMBEDDINGS,
    developer_id_map=DEVELOPER_ID_MAP,
    expected_developers=ACTIVE_DEVELOPERS_LIST,
    train_data=TRAIN_DATAFRAME,
    device="cuda",
    similarity_prediction_weight=WEIGHT_FOR_SIMILARITY_PREDICTION,
    time_decay_factor=OPTIMIZED_TIME_DECAY_FACTOR,
    direct_assignment_score=CONTRIBUTION_POINT_FOR_DIRECT_ASSIGNMENT,
    contribution_score=CONTRIBUTION_POINT_FOR_COMMITS_PR,
    discussion_score=CONTRIBUTION_POINT_FOR_DISCUSSION,
)

# Get Recommendation
triagerx_pipeline.get_recommendation(
        "Bug Title\nBug Description",
        k_dev=TOP_K_DEVELOPERS,
        k_rank=MAXIMUM_SIMILAR_ISSUES_TO_BE_CONSIDERED,
        similarity_threshold=THRESHOLD_FOR_ISSUE_SIMILARITY,
    )
```
This is the basic setup to use TriagerX. A complete demo for Openj9 dataset is provided in [`triagerx/trainer/demo.py`](triagerx/trainer/demo.py) script.

## Optimizing Hyperparameters for IBR
IBR 


## Baseline Reproduction
We reproduce literature baselines (LBT-P and DBRNN-A) as the source codes are not publicly available. The following steps explain how the baselines can be reproduced.

### LBT-P
Firstly, we distill RoBERTa-large using Patient Knowledge Distillation. The model can be distilled with the following command. The example below demonstrates distillation for the Google Chromium dataset.
```bash
python reproduction/lbtp_distillation.py \
--dataset_path data/deeptriage/google_chrome/deep_data.csv \
--model_weights_path output/lbtp_gc_base.pt
```
Once distillation is done, the classifier can be trained with the following command:
```bash
python reproduction/train_lbtp.py \
        --dataset_path data/deeptriage/google_chrome/classifier_data_20.csv \
        --embedding_model_weights output/lbtp_gc_base.pt \
        --block 9 \
        --output_model_weights output/lbtp_gc.pt \
        --run_name lbpt_gc \
        --wandb_project wandb_project_name
```
### DBRNN-A
We reproduced DBRNN-A following this [repository](https://github.com/hacetin/deep-triage/tree/master) and the paper. Since it is based on Tensorflow, we recommend creating a new environment using this [requirements file](reproduction/dbrnna/requirements.dbrnna.yml) similar to our project.

Once the environment is created and activated, run the following script

```bash
python reproduction/dbrnna/main.py
```

## Build Docker Image
To build the Docker image for Triager X, run the following command:

```shell
docker build -t triagerx .
```

## Load Docker Image
To build the Docker image for Triager X, run the following command:

```shell
docker load -i triagerx.tar
```

## Run Docker Container
To run the Docker container on CPU, use the following command:
### CPU
```shell
docker run --rm -p 8000:80 --name triagerx triagerx
```

To run the Docker container with GPU support, use the following command:
### GPU
```shell
docker run --gpus all --rm -p 8000:80 --name triagerx triagerx
```

## Example API Request
To get component and developer recommendations for a GitHub issue, make a POST request to the `/recommendation` endpoint. Here is an example using `curl`:

```shell
curl -X 'POST' \
  'http://127.0.0.1:8000/recommendation' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "issue_title": "Issue Title from GitHub",
  "issue_description": "Issue Description from GitHub"
}'
```

## Example API Response
The API will respond with a JSON object containing the recommended components and developers. Here is an example response:

```json
{
  "recommended_components": [
    "comp:vm",
    "comp:gc",
    "comp:test"
  ],
  "recommended_developers": [
    "pshipton",
    "keithc-ca",
    "babsingh"
  ]
}
```

## Swagger UI

You can also invoke the endpoint with Swagger UI.
To access the UI for using the API or reading the documentation,
navigate to the following address once the container is up and running:

```
http://127.0.0.1:8000/docs
```

## Usage
1. **Build the Docker Image**: Follow the instructions in the "Build Docker Image" section.
2. **Run the Docker Container**: Follow the instructions in the "Run Docker Container" section.
3. **Make API Requests**: Use the example API request to get recommendations for your GitHub issues.