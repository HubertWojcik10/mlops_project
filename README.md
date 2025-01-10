# Image Classification using the Fashion-MNIST Dataset

## Project Description

Fashion-MNIST (FMNIST) is an image dataset consisting of clothing articles from Zalando. It consists of 60,000 images in the training set and 10,000 in the test set. Each image is a 28x28 grayscale image and therefore results in 784 individual pixels per image which are represented as features within the dataset. Additionally, the true label corresponding to each image is known and represented as a column in the dataset. Similar to MNIST, it serves as a benchmark to many modern ML models, but also introduces more complexity at the same time in terms of the images.

The overall goal of the project is to build a trustworthy CNN model by clearly defining and maintaining an MLOps pipeline. For easy and organized collaboration, we will fork the provided MLOps cookiecutter template on GitHub, and then to achieve reproducibility, we aim to utilize Docker.

## Project Structure (TBD)

The directory structure of the project (as per template) looks like this:
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
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
