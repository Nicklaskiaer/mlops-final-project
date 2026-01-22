## Cloud setup action logs
gcloud auth login
gcloud services enable artifactregistry.googleapis.com run.googleapis.com

gcloud artifacts repositories create infant-cry-repo \
    --repository-format=docker \
    --location=europe-west1 \
    --description="Docker repository for Infant Cry Project"

gcloud auth configure-docker europe-west1-docker.pkg.dev

  output:
  Adding credentials for: europe-west1-docker.pkg.dev
  After update, the following will be written to your Docker config file located at [/Users/nicklaskiaer/.docker/config.json]:
  {
    "credHelpers": {
      "europe-west1-docker.pkg.dev": "gcloud"
    }
  }

  Do you want to continue (Y/n)?  y

  Docker configuration file updated.

docker tag infant_cry_api:v1 europe-west1-docker.pkg.dev/dev-lane-484207-e9/infant-cry-repo/infant_cry_api:v1
docker push europe-west1-docker.pkg.dev/dev-lane-484207-e9/infant-cry-repo/infant_cry_api:v1

gcloud run deploy infant-cry-api \
    --image europe-west1-docker.pkg.dev/dev-lane-484207-e9/infant-cry-repo/infant_cry_api:v1 \
    --region europe-west1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --port 8080

## docker:
(normal build)
docker build -f dockerfiles/api.dockerfile -t infant_cry_api:v1 .

(mac os build, for cloud deployment)
docker build \
  --platform linux/amd64 \
  -f dockerfiles/api.dockerfile \
  -t europe-west1-docker.pkg.dev/dev-lane-484207-e9/infant-cry-repo/infant_cry_api:v1 .

docker run -p 8000:8000 infant_cry_api:v1



# Infant Cry Classification Using Lightweight HuBERT

## Project Overview

The goal of this project is to develop a supervised learning system capable of classifying infant emotional states based on audio recordings of infant cries. Accurate cry classification has potential applications in healthcare, parental support systems, and early child development monitoring.

This project explores modern transformer-based speech models that operate directly on raw audio waveforms, avoiding handcrafted feature extraction techniques such as mel-spectrograms. In particular, we focus on the HuBERT architecture and its lightweight variant, which enable efficient learning from limited labeled data.

---

## Dataset

We use the **Infant Cry Dataset** available on Kaggle:

- **Dataset link:** https://www.kaggle.com/datasets/sanmithasadhish/infant-cry-dataset/data
- **Modality:** Audio (raw waveform)
- **Number of samples:** ~800 audio clips
- **Number of classes:** 8 infant emotion labels
- **Class distribution:** Approximately 100 samples per class

The dataset size is relatively small, motivating the use of pretrained models and careful evaluation strategies.

---

## Modeling Approach

### Baseline Models

To establish a performance baseline, we consider simple heuristic models:

- **Random guessing**
- **Majority class prediction**, where the model always predicts the most frequent label

These baselines serve as lower bounds for model performance.

---

### Proposed Models

We investigate transformer-based speech representation models:

- **HuBERT (Hidden-Unit BERT)**  
  A state-of-the-art self-supervised transformer model that learns rich speech representations directly from raw audio.

- **Lightweight HuBERT**  
  A computationally efficient variant of HuBERT designed to reduce model complexity while retaining strong performance.

Given the limited dataset size, we fine-tune a pretrained Lightweight HuBERT model rather than training from scratch.

---

## Training Strategy

- **Learning paradigm:** Supervised learning  
- **Task:** Multi-class audio classification (8 classes)  
- **Input:** Raw audio waveforms  
- **Output:** Infant emotion label  

To robustly assess model performance, we employ a **two-level (nested) cross-validation** strategy, enabling reliable evaluation despite the limited amount of data.

---

## Evaluation Metrics

Model performance is evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  

These metrics provide a comprehensive view of classification performance, particularly in the presence of potential class imbalance.

---

## Project Workflow

1. Select dataset and model architecture  
2. Create project repository  
3. Upload project description as part of `README.md`  
4. Implement baseline models  
5. Fine-tune Lightweight HuBERT on the dataset  
6. Evaluate performance using nested cross-validation  
7. Analyze results and compare against baselines  

---

## Expected Outcomes

This project aims to demonstrate that pretrained transformer-based audio models, even in lightweight form, can effectively classify infant emotional states from cry audio despite limited labeled data. The results will provide insight into the feasibility of deploying such models in real-world, resource-constrained environments.




# project

Final project

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
