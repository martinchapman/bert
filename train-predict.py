# Train

import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score


from util import Util


Util.set_seed(23)


# Specify example
example = 'examples/biomarker-classification'
with open(example + '/config.json', 'r') as file:
    config = json.load(file)


with open(example + '/' + config['train_dataset'], 'r') as file:
    data = json.load(file)
# Get source data into format for training
labels = sorted(set([label for row in data for label in row['labels']]))
data = [
    ({'text': entry['text']} | {label: label in entry['labels'] for label in labels})
    for entry in data
]
dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.2)


# Check class balance
print(Util.class_balance(dataset['train'], labels))


tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
model = AutoModelForSequenceClassification.from_pretrained(
    config['model_name'],
    num_labels=len(labels),
    problem_type='multi_label_classification',
    id2label={i: label for i, label in enumerate(labels)},
    label2id={
        label: i for i, label in enumerate(labels)
    },  # Possibly unnecessary, but feels like good practice to provide mappings (e.g. if trained model shared)
)


# Freeze encoder layers
for param in model.bert.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True


def tokenize(element):
    tokens = tokenizer(
        element['text'], padding='max_length', truncation=True, max_length=128
    )
    if len(list(element.keys())) > 1:  # if we have features, format them
        tokens['labels'] = [float(element[label_name]) for label_name in labels]
    return tokens


# Could batch here for efficiency, but not batching allows for a more intuitive tokenize function and assume training data is of a relatively small size (Simplification 1)
dataset = dataset.map(tokenize, remove_columns=dataset['train'].column_names)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


training = TrainingArguments(
    './output',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=config['learning_rate'],
    per_device_train_batch_size=16,  # Alter for hardware
    per_device_eval_batch_size=16,
    num_train_epochs=config['epochs'],
    weight_decay=config['weight_decay'],
    load_best_model_at_end=True,
    metric_for_best_model=config['metric_for_best_model'],
)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {
        'f1': f1_score(
            y_true=labels,
            y_pred=Util.reformat_predictions(predictions, config['threshold']),
            average='weighted',
        )
    }


trainer = Trainer(
    model=model,
    args=training,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)
trainer.train()


metrics = trainer.evaluate()
print(metrics)
with open(example + '/metrics.txt', 'w') as file:
    file.write(json.dumps(metrics))


# Predict

with open(example + '/' + config['prediction_dataset'], 'r') as file:
    prediction_data = json.load(file)
prediction_dataset = Dataset.from_list(
    [{'text': entry['text']} for entry in prediction_data]
)
prediction_dataset = prediction_dataset.map(
    tokenize, remove_columns=prediction_dataset.column_names
)
prediction_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


predictions = trainer.predict(prediction_dataset)
formatted_predictions = []
for prediction_index, prediction in enumerate(
    Util.reformat_predictions(predictions.predictions, config['threshold'])
):
    predicted_labels = [
        labels[label_index]
        for label_index, value in enumerate(prediction)
        if value == 1
    ]
    formatted_predictions.append(
        {
            'text': prediction_data[prediction_index]['text'],
            'predicted_labels': predicted_labels,
        }
    )
print(json.dumps(formatted_predictions, indent=2))
with open(example + '/predict.txt', 'w') as file:
    json.dump(formatted_predictions, file, indent=2)
