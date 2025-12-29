## Learning-to-rank algorithms

This repository contains example implementations of deep learning and gradient boosting based algorithms for the task of learning to rank a set of candidates to find for a user an approximate rating of a movie based on what the user has rated before.

### Models

The following models are tested within this repository:
 - LGBMRanker ([Website](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRanker.html))
 - DeepFM ([Paper](https://arxiv.org/pdf/1703.04247))
 - DCNv2 ([Paper](https://arxiv.org/pdf/2008.13535))

### Dataset

The repository contains a dataset loader for the Movie-lens 100K dataset accesible [here](https://grouplens.org/datasets/movielens/100k/). The contents can be extracted into the `data` folder and the code should work without any changes.

### Examples

To train the model with `DeepFM`, run:

```sh
uv run -m src.main --model deepfm
```

and with DCNv2:

```sh
uv run -m src.main --model dcnv2
```

The model parametrs can be configured by changing the files within the `config` folder.