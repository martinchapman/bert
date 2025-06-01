# bert

BERT fine-tuning pipeline

## Getting Started

1. Create a virtual environment

    ```
    pip install -m venv .venv
    ```

2. Install packages

    ```
    pip install -r requirements.txt
    ```

3. Specify an existing example (checking any specific installation requirements within the relevant subfolder), or create and specify a new example, within [train-predict.py](train-predict.py), e.g.:

    ```
    example = 'examples/biomarker-classification'
    ```

## Usage

Run [train-predict.py](train-predict.py).

## Simplifications

1. Only handle training data that is of a relatively small size.

2. Only handle training data of the form:

    ```
    {
        text: "foo",
        labels: ["bar", "baz"]
    }
    ```

3. Use Hugging Face libraries to abstract common pipeline tasks.

4. Freeze all but the classifier layer of the given model.

5. Threshold used to determine positive labels from probabilities is uniformly applied (rather than having a different threshold per label). 

6. Use f1 score alone to provide a summary of performance.

## Examples

- [Biomarker Classification](examples/biomarker-classification/README.md): Fine-tuning BERT for biomarker classification.
