# TriagerX

TriagerX is a novel triaging framework designed to streamline the process of handling GitHub issues. By analyzing the title and description of a GitHub issue, and using two rankers (1) TriagerX Content-based Ranker (CBR), (2) TriagerX Interaction-based Ranker (IBR), TriagerX recommends the most appropriate developers to address the issue. This tool can significantly reduce the time and effort needed to manage and assign issues in software projects. In this README, we provide extensive guidelines on how to get TriagerX up and running.


## Environmental Setup
Create a new `conda` environment by executing the following command:

```bash
conda env create -f environment.dev.yml
```
Activate the environment by:

```bash
conda activate triagerxdev
```

## Preparing the Datasets
Download the bug datasets compressed in this zip file: [**data.zip - Google Drive**](https://drive.google.com/file/d/1TGnCJvQb04W8pae0pfm4K3_fu0ySkvI1/view?usp=sharing)

Once the compressed zip is downloaded, **extract it to the root of the project directory**. The directory structure should be something similar to this:
```bash
data/
│   ├── google_chrome/
│   ├── mozilla_core/
│   ├── mozilla_firefox/
│   ├── openj9/
│   │   ├── issue_data/
│   │   └── openj9_bug_data.csv
│   └── typescript/
│       ├── issue_data/
│       └── ts_bug_data.csv
```
The data/datasets inside `google_chromium`, `mozilla_core`, and `mozilla_firefox` directories are from [literature](https://dl.acm.org/doi/abs/10.1145/3297001.3297023) and collected from [this repository](https://github.com/hacetin/deep-triage). This can only be used with the TriagerX CBR.

`openj9` and `typescript` directories contain the datasets we prepared. Each of these directories have complete github issues data with interaction events in `json` format. While to train or run TriagerX, no further data preparation is required as we are providing it, issue data can be downloaded using Github API from any repository using the following pipeline:

```bash
python util/issue_dump.py --path data/openj9 --owner openj9 --repo openj9
```
However, a github token is required. We set the token on `.env` with `GH_TOKEN` key.

Once the issue data is downloaded, we can use this data for TriagerX IBR component and prepare it for TriagerX CBR training. Similarly, `*_bug_data.csv` is used for training TriagerX CBR where we have issue summaries, descriptions and the bug fixers. This CSV datasets can be prepared by using [`util/make_dataset.py`](util/make_dataset.py) script. We just need to refer the JSON directory root (e.g., `data/openj9/issue_data`) and the output CSV path. This step is also not required if the provided data is used.


## Training
_All the models are trained on NVIDIA A100 GPU. We recommend to use the same. While TriagerX can also be trained on V100 GPUs, the large variants of DeBERTa or RoBERTa may require more space when trained with CNN classifiers._

We provide seperate training pipelines for each dataset. All the training scripts can be found in [`training/developer`](training/developer) directory. Training scripts require training configs defined in `yaml` files. Each training pipeline on each dataset can run on 3 different types of configuration (e.g., **triagerx, cnn-transformer, fcn-transformer**). Configuration files are provided for all combinations. All of our training configs can be found under [`training/training_config`](training/training_config).

Some configuration parameters are described below:

```yaml
use_description: true
# Boolean flag indicating whether to use descriptions within the model. When False, the model will be trained with only bug title/summary.

base_transformer_models: 
 - "microsoft/deberta-base"
 - "roberta-base"
# List of pre-trained transformer model keys from the HuggingFace repository to be used as base models for the ensemble. Multiple models to be used only when we are training TriagerX configuration.

unfrozen_layers: 3
# Number of layers of PLM to unfreeze during training. Unfrozen layers will be fine-tuned while the rest remain frozen. -1 means all layers will be trained.

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
```shell
python training/developer/developer_training_openj9.py \
        --config training/training_config/openj9/triagerx.yaml \
        --dataset_path ~absolute-path/data/openj9/openj9_bug_data.csv \
        --seed 42
```
_If `wandb` is used, a valid `WANDB_API_KEY` is required in the environment variable. It can be turned off manually from the training script by setting:_
```python
log_manager = EpochLogManager(wandb_config=None)
```
In case, `triagerx` modules can not be detected by the interpreter, please check for this specific line in the scripts and make sure you append the correct repository root to the system path.
```python
sys.path.append("../triagerX")
```

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
This is the basic setup to use TriagerX. A complete demo for Openj9 dataset is provided in [`triagerx/trainer/demo.py`](triagerx/trainer/demo.py) script. **Please note that, you have to provide the correct saved model path (`developer_model_weights`) from the training step.** We provide the trained weights for Openj9 here that can be used directly [**Openj9 - Trained Weights (50 Developers)**](https://drive.google.com/file/d/1gYbOboGUjVqCUEDNWdHC1GBR1qcrfgn5/view?usp=sharing).

For TriagerX IBR, we are currently using [**all-mpnet-base-v2**](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) but any supported models by [SentenceTransformer](https://www.sbert.net/index.html) that can utilize cosine-similarity score function can be used.

## Optimizing Hyperparameters for IBR
TriagerX IBR can be optimized through Grid Search with the following script. The process is to use different combination of hyperparameters and see which combination works the best on a specific test data. a numeric list of ranges for different parameters is passed and run on TriagerX pipeline using a specific test data. All results using all combination are then saved to a CSV. The best performing hyperparaneters are then used for final use.

For example, in our grid search process, we can use hyperparameter ranges by defining them in a dictionary.
```json
parameter_ranges = {
    "similarity_prediction_weight": [0.50, 0.65],
    "time_decay_factor": [-0.01, 0.01],
    "direct_assignment_score": [0.2, 0.5],
    "contribution_score": [0.5, 1.5],
    "discussion_score": [0.1, 0.2],
    "similarity_threshold": [0.2, 0.5],
}
```
Grid Search is implemented in this script: [`triagerx/trainer/grid_search.py`](triagerx/trainer/grid_search.py). We recommend uncommenting `@lru_cache()` in `triagerx/system/triagerx.py` file before starting grid search to speed up the process.

## Testing
We evaluated TriagerX on unseen issues from Openj9 repository using 17 active developers/80 issues. These issues are saved in `JSON` format in the `test_issues_dump/` directory. To run the evaluation:

```bash
python test/eval_developer_predictions.py 
```
You should get a result like the following for Top-3 developer predictions:
```
Total correct predictions: 34 out of 51, Acc: 0.6666666666666666
Total correct predictions with discussion: 66 out of 80, Acc: 0.825
```

## Statistical Test
To run statistical test, first update the configuration file. For example, we provided the file for Openj9 in `training/training_config/openj9/developer/stat_test.yaml`. The file should be updated with correct trained model configuration as the following:


```yaml
use_description: true
model1_transformer_models: 
 - "microsoft/deberta-base"
 - "roberta-base"
model2_transformer_models:
  - "roberta-large"
model1_unfrozen_layers: 3
model1_num_classifiers: 3
model2_unfrozen_layers: 1
model2_num_classifiers: 1
model1_key: "triagerx"
model2_key: "lbtp"
val_size: 0.1
test_size: 0.1
model1_dropout: 0.2
model2_dropout: 0.2
batch_size: 10
model1_max_tokens: 256
model2_max_tokens: 256
early_stopping_patience: null
topk_indices:
  - 1
  - 3
  - 5
  - 10
  - 20
model1_path: "./openj9_triagerx_u3c3_50_classes_last_dev_seed42.pt"
model2_path: "./lbtp_openj9_block9.pt"

```
Then run:
```bash
python training/developer/statistical_test.py --config training/training_config/openj9/developer/stat_test.yaml --dataset_path data/openj9/openj9_08122024.csv --seed 42
```


## Running the API
TriagerX framework provides fast & clean deployment through FastAPI. To make the recommendation API up and running for Openj9 (Can also be used for other supported datasets) follow the provided steps:

1. Move the saved weights for TriagerX CBR model under `app/saved_states`.
2. Copy all github issues from `data/openj9/issue_data` to `app/app_data/issue_data`.
3. Copy the train csv file from `data/openj9/openj9_train.csv` to `app/app_data/openj9_train.csv`.
4. Set configurations for the app in [`app/config/triagerx_config.yaml`](app/config/triagerx_config.yaml) file. It is self explanatory.
5. Once the configuration is set, run the API by following command:
    ```shell
    uvicorn main:app --host 0.0.0.0 --port 8000
   ```
6. The first boot may take some time to initialize a few things. When the app startup is complete, the API can be used by:
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/recommendation' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{"issue_title": "Issue Title from GitHub", "issue_description": "Issue Description from GitHub"}'
   ```
7. The API will respond with a JSON object containing the recommended components and developers. Here is an example response:
    ```json
    {
      "recommended_developers": [
        "pshipton",
        "keithc-ca",
        "babsingh"
      ]
    }
    ```
You can also invoke the endpoint with Swagger UI dashboard. To access the UI for using the API or reading the documentation, navigate to the following address once the API is up and running: http://127.0.0.1:8000/docs

## Docker Containerization
### Build Docker Image
The framework along with the recommendation API can also be dockarized and run inside a docker container. To build the Docker image for Triager X when all the required configuration is done from above, run the following command:

```shell
docker build -t triagerx .
```

### Run Docker Container
To run the Docker container on CPU, use the following command:
#### CPU
```shell
docker run --rm -p 8000:80 --name triagerx triagerx
```

To run the Docker container with GPU support, use the following command:
#### GPU
```shell
docker run --gpus all --rm -p 8000:80 --name triagerx triagerx
```

_Please note that the Docker container is currently intended for `ppc64le` platform, as per the requirements of our industrial partner but it should be working in other platforms with just changing the conda repository to a suitable one from here: https://repo.anaconda.com/miniconda/ in the Dockerfile._

## Baseline Reproduction
We reproduce literature baselines (LBT-P and DBRNN-A) as the source codes are not publicly available. The following steps explain how the baselines can be reproduced.

### LBT-P
Firstly, we distill RoBERTa-large using Patient Knowledge Distillation. The model can be distilled with the following command. The example below demonstrates distillation for the Google Chromium dataset. LBT-P can be trained with TriagerX's environment as there is no exclusive dependencies.
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
We reproduced DBRNN-A following this [repository](https://github.com/hacetin/deep-triage/tree/master) and the original paper. Since it is based on Tensorflow, we recommend creating a new environment using this [requirements file](reproduction/dbrnna/requirements.dbrnna.yml) similar to our project.

Once the environment is created and activated, run the following script

```bash
python reproduction/dbrnna/main.py
```
### Additional Baselines
We also evaluated some additional baselines like NCGBT (https://github.com/wormedleaf/NCGBT/tree/master) and PCG (https://gitee.com/papercodefromasexd/pcg). However, this methods underperformed on our datasets, hence, is not reported in the paper. One likely reason is that NCGBT heavily relies on CC fields, which are missing in our datasets.

**Table: Results on different datasets using NCGBT**

| Dataset         | Approach | Top-1  | Top-3  | Top-5  | Top-10 | Top-20 |
|----------------|----------|--------|--------|--------|--------|--------|
| TS             | NCGBT    | 0.0503 | 0.1661 | 0.2648 | 0.4255 | 0.7065 |
| Openj9         | NCGBT    | 0.0353 | 0.1191 | 0.1700 | 0.2848 | 0.4437 |
| Google Chromium| NCGBT    | 0.0036 | 0.0098 | 0.0148 | 0.0264 | 0.0477 |


**Table: Results on Google Chromium using PCG**

| Dataset          | Approach | Top-1  | Top-3  | Top-5  | Top-10 | Top-20 |
|------------------|----------|--------|--------|--------|--------|--------|
| Google Chromium  | PCG      | 0.0076 | 0.0102 | 0.0165 | 0.0293 | 0.0560 |

To rerun these baselines, download the respective zip file and follow the attached README for each zip and run `main.py` files. It includes all the processed data and steps to reproduce.

NCGBT: https://drive.google.com/file/d/1LUTe3VQDraM7awE4TKWqe3bUUIStc4R4/view?usp=sharing

PCG: https://drive.google.com/file/d/1Afqj5n5BjrJFXdXwTMsBgP9ZG0IaJjO-/view?usp=sharing
## Additional Materials
Some additional notebooks are provided under [`notebook`](notebook) directory that were used for evaluation/exploratory analysis or visualization.