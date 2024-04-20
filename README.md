# ArrowTextClassifier

ArrowTextClassifier is a Python package for text classification tasks, offering functionalities to train, summarize, and classify text using convolutional neural network (CNN) architecture.

## Installation

You can install ArrowTextClassifier via pip:

```bash
pip install ArrowTextClassifier
```

## How it Works

ArrowTextClassifier implements a convolutional neural network (CNN) architecture for text classification. It tokenizes input text, embeds the tokens, applies convolutional filters over the embedded tokens to extract features, and then classifies the text into predefined categories.

## Usage

### Training

To train a text classification model, you can utilize the `train_model` method provided by the `Model` class:

```python
from ArrowTextClassifier import Model

model = Model(name="your_model_name")
model.train_model(dataset)
```

#### How to make a dataset

To make your own custom dataset for training you need to create a parquet file with the following format:

*Example Parquet File*

```json
{"label":"normal","example":"Hey there!"}
{"label":"normal","example":"Hi!"}
{"label":"toxic","example":"You suck!"}
```

After you have created the parquet file with the data in the format above, you can provide to the dataset to start training the model.

### Summarization

To summarize a trained model, you can use the `summarize` method:

```python
model.summarize(
    model_path="path_to_your_model",
    hyperparams_path="path_to_hyperparameters_file",
    vocabulary_path="path_to_vocabulary_file",
    modelSummary_write_path="path_to_write_model_summary"
)
```

### Classification

For classifying text using the trained model:

```python
result = model.classify(
    model_path="path_to_your_model",
    hyperparams_path="path_to_hyperparameters_file",
    text="your_input_text",
    vocabulary_path="path_to_vocabulary_file"
)
print(result)
```

## Getting Started

This package provides tools for text classification tasks. You can explore and customize it according to your requirements. Refer to the documentation for detailed usage instructions.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For any questions or feedback, please contact technologypower24@gmail.com or you can contact me at discord - techpowerb.