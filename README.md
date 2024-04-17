# ArrowTextClassifier

ArrowTextClassifier is a simple text classification tool written in pytorch that allows you to train, summarize, and use text classification models for various tasks.

## How it Works

ArrowTextClassifier uses a convolutional neural network (CNN) architecture for text classification. It tokenizes input text, embeds the tokens, applies convolutional filters over the embedded tokens to extract features, and then classifies the text into predefined categories.

## Model Architecture

The CNN model consists of an embedding layer followed by multiple convolutional layers with different filter sizes. Max-pooling is applied over the output of convolutional layers to extract the most relevant features. Finally, a fully connected layer with dropout is used to classify the text into different categories.

## Training Script

To train the model, you can use the provided `train.py` script. You have two options for training:
1. Using the provided dataset in the `classification` folder.
2. Using your own custom dataset.

### Using Given Dataset
```bash
python src/train.py
```

This will start training the model with the provided dataset. You can configure the parameters in the `train.py` script if needed.

### Using Your Own Custom Dataset
1. Prepare your dataset in Parquet format with each example containing a label and text.
2. Add your dataset to the `classification` folder.
3. Change the dataset path at line 100 of `train.py` to point to your custom dataset.
4. Run `python train.py` to start training with your custom dataset.

*Example Parquet File*

```json
{"label":"normal","example":"Hey there!"}
{"label":"normal","example":"Hi!"}
{"label":"toxic","example":"You suck!"}
```

After you have finished collecting the training data, you can use the following command below

```bash
python src/train.py <path/to/your/dataset>
```

## Summarize Script

We also provide a summarization script, `summarize_model.py`, which summarizes the trained model. It provides information about the vocabulary, hyperparameters, and the model architecture.

```bash
python src/summarize_model.py
```
You need to provide the path to the trained model, hyperparameters file, and vocabulary file as arguments.

## Classifier Script

To classify text using the trained model, you can use the `src/classify_text.py` script. Provide the paths to the trained model, hyperparameters file, and vocabulary file as arguments.

```bash
python src/classify_text.py
```

## Getting Started

A pretrained model has been provided in the `pretrained_model` directory to help you get started. Keep in mind that this is just a test model and is not recommended for large-scale use, only for testing purposes.

This is a simple model that can be customized according to your needs. Feel free to explore and make any modifications you require.
We have also made our own colab [notebook](https://colab.research.google.com/drive/1fGDLICkctfdpTgLoh_Bouv-NY-q-kdlQ?usp=sharing)

---
This project was created by Bhargav230m and is provided under the MIT License license. For any questions or feedback, please contact technologypower24@gmail.com.