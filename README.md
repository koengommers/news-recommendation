# Personalized News Recommendation

This repository contains PyTorch implementations of several personalized news recommendation methods, created for my MSc thesis in Artificial Intelligence at University of Amsterdam.

Many news recommendation models follow the same general architecture with similar components: a news encoder, user encoder and click predictor. That is the perspective this repository relies on. The individual components are available as modules and through configuration they are combined to form the model.

This repository relies heavily on [Hydra](https://hydra.cc/) for configuration, so it is recommended to familiarize yourself with it.

## Available models

| Name   | Paper                                                                                                               | Notes                               |
|--------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| NAML   | [Neural News Recommendation with Attentive Multi-View Learning](https://www.ijcai.org/proceedings/2019/536)         |                                     |
| NRMS   | [Neural News Recommendation with Multi-Head Self-Attention](https://www.aclweb.org/anthology/D19-1671)              |                                     |
| TANR   | [Neural News Recommendation with Topic-Aware News Representation](https://www.aclweb.org/anthology/P19-1110)        |                                     |
| HieRec | [HieRec: Hierarchical User Interest Modeling for Personalized News Recommendation](http://arxiv.org/abs/2106.04408) | Only user encoder + click predictor |
| MINER  | [MINER: Multi-Interest Matching Network for News Recommendation](https://aclanthology.org/2022.findings-acl.29)     | BERT news encoder performs poorly   |


## Getting started

After cloning, install dependencies using [Poetry](https://python-poetry.org/):

    poetry install

By default, a sampled subset of the MIND-large dataset is used. This is because the original dataset does not contain test labels. Either you need to use the MIND-large dataset and disable evaluation on test split (through adding `data=mind_large eval_splits=[dev]` to your command) or you can sample the data through:

    poetry run python src/sample_data.py

Training a model (e.g. NRMS):

    poetry run python src/train_recommender.py +model=nrms

Output can be found in the `outputs/` directory

## Custom model configurations

It is possible to combine model components to create a custom model. A new entry could be added to the  `conf/model/` directory, or it could be done through command line arguments. For example, NRMS model with TANR user encoder:

    poetry run python src/train_recommender.py +model=nrms model/user_encoder=additive_attention

There are some pre-made presets for Hierarchical User Interest Modeling (from HieRec) and Multi-Interest User Modeling (from MINER). Example for using NRMS with Multi User-interest:

    poetry run python src/train_recommender.py +model=nrms +options=multi_interest

Note: you are responsible for ensuring the necessary features for each component are selected.

## Acknowledgements

- Credits to all the authors of the papers
- Microsoft News Dataset (MIND), see [https://msnews.github.io/](https://msnews.github.io/).
- NAML, NRMS and TANR are adapted from implementation of yusanshi, see [https://github.com/yusanshi/news-recommendation](https://github.com/yusanshi/news-recommendation).
