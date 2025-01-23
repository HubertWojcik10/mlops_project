# Image Classification using the Fashion-MNIST Dataset

## Project Description

Fashion-MNIST (FMNIST) is an image dataset consisting of clothing articles from Zalando. It consists of 60,000 images in the training set and 10,000 in the test set. Each image is a 28x28 grayscale image and therefore results in 784 individual pixels per image which are represented as features within the dataset. Additionally, the true label corresponding to each image is known and represented as a column in the dataset. Similar to MNIST, it serves as a benchmark to many modern ML models, but also introduces more complexity at the same time in terms of the images.

The overall goal of the project is to build a trustworthy CNN model by clearly defining and maintaining an MLOps pipeline. For easy and organized collaboration, we will fork the provided MLOps cookiecutter template on GitHub, and then to achieve reproducibility, we aim to utilize Docker.

## Project Structure

The directory structure of the project looks like this:
```txt
├── .dvc/
│   ├── .gitignore
│   └── config
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│   │   └── unnittests.yaml
├── configs/                    # Configuration files
│   ├── .gitkeep
│   ├── api_cloudbuild.yaml
│   ├── cloudbuild.yaml
│   ├── config.yaml
│   ├── config_cpu.yaml
│   ├── sweep.yaml
│   └── vertex_ai_train.yaml
├── data/                     # Data directory
│   └── processed/
│   │   ├── test_data.pkl.dvc
│   │   └── train_data.pkl.dvc
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── models/                   # Trained models
│   └── .gitkeep
├── reports/                  # Reports
│   ├── README.md
│   ├── report.py
│   └── figures/
│   │   ├── bucket.png
│   │   ├── buckets_overview.png
│   │   ├── build.png
│   │   ├── fashion_mnist_bucket.png
│   │   ├── overview.png
│   │   ├── registry.png
│   │   ├── structure.png
│   │   ├── wandb.png
│   │   └── wandbours.png
├── src/                      # Source code
│   ├── mlops_project/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── model.py
│   │   └── train.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .dvcignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
