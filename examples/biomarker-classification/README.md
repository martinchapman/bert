# Biomarker Classification

Fine-tuning BERT for biomarker classification.

An example [config](config.json), with corresponding sample [training metrics](metrics.txt) and [prediction output](predict.txt), is provided.

## Getting Started

1. Init submodule

    ```
    git submodule update --init --recursive
    ```

## Assumptions

1. Training and validation data all comes from [train.json](https://github.com/londonaicentre/nlptakehome/blob/9411e7752e0bfddb827d703ad782d4d89f50782e/data/train.json).

2. More significant steps to deal with the small class imbalance present, beyond those already taken (e.g. explicit weight calculations fed into the training process), is out of scope.

3. `bert-base-cased` is suitable, as it balances performance and accuracy particularly with smaller dataset like this one.

4. This is a multi-label classification task (as opposed to discreet sets of labels).

5. Although input text is longer, 128 max length (with truncate) during tokenization suitably captures relationships while retaining performance.

6. A threshold of 0.5 is sufficient (see [Simplifications](../../README.md#Simplifications)).

7. Weighted f1 helps to offset any class imbalance.

### Hyperparameter assumptions

Hyperparameters are designed to be experimented with to optimise. 
For now, the following assumptions have been made:

1. Learning rate: Small weight increments suitable due to size of task.

2. Epochs: A typical standard values of 3 is suitable.

3. Weight decay: Standard coefficient used is suitable.

4. Best model metric: f1 as 'textbook' metric for diagnosis is suitable.