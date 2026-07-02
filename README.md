# Deep-Learning-Next-Word-Prediction-Engine
# Next Word Prediction with Bidirectional LSTM

A deep learning model that predicts the next word in a sequence, trained on the text of *Sherlock Holmes*. Built with TensorFlow/Keras using a stacked Bidirectional LSTM architecture.

## Overview

This project trains a language model to predict the next word given a sequence of preceding words. It uses classic NLP preprocessing (tokenization, n-gram sequence generation, padding) combined with an embedding + Bidirectional LSTM network, and includes a text generation function for producing multi-word continuations from a seed phrase.

## Dataset

The model is trained on the *Sherlock Holmes* text corpus. The raw text is lowercased, split into non-empty lines, and used to build a vocabulary via Keras' `Tokenizer`.

## How It Works

1. **Preprocessing**
   - Text is lowercased and split into sentences (non-empty lines).
   - A `Tokenizer` (with an `<unkn>` out-of-vocabulary token) is fit on the sentences to build the vocabulary.
   - Each sentence is converted into a sequence of token IDs, and progressively growing n-gram subsequences are generated (e.g., for a 5-word sentence, sequences of length 2, 3, 4, 5 are created). This teaches the model to predict the next word at every position.
   - Sequences are left-padded (`padding='pre'`) to a uniform length based on the longest sequence.
   - Each padded sequence is split into input (`X`, all tokens except the last) and target (`y`, the last token), and `y` is one-hot encoded.

2. **Model Architecture**
   - `Embedding` layer (256-dim word embeddings)
   - `Bidirectional LSTM` (256 units, returns sequences) + `Dropout` + `LayerNormalization`
   - `Bidirectional LSTM` (100 units, final output)
   - `Dropout`
   - `Dense` output layer with softmax activation over the vocabulary

3. **Training**
   - Loss: categorical cross-entropy
   - Optimizer: Adam
   - Metrics: accuracy, top-3 categorical accuracy
   - Early stopping on validation accuracy (patience = 5, restores best weights)
   - 30 epochs max, batch size 64, 20% validation split

4. **Text Generation**
   - Given a seed phrase, the model predicts the next word iteratively.
   - Supports two decoding strategies:
     - `greedy`: always picks the highest-probability word
     - `sampling`: applies temperature-scaled sampling for more varied, less repetitive output

## Requirements

```
numpy
tensorflow
matplotlib
requests
```

Install with:

```bash
pip install numpy tensorflow matplotlib requests
```

## Usage

Run the notebook (`NextWordPred.ipynb`) cell by cell. It will:

1. Download and preprocess the training text.
2. Tokenize and build training sequences.
3. Build and train the LSTM model.
4. Plot training/validation loss and accuracy.
5. Generate sample text from a seed phrase.

Example generation call:

```python
generated_story = generate_text(
    seed_text="go",
    next_words=4,
    model=model,
    tokenizer=tokenizer,
    max_len=max_len,
    temperature=0.7,
    mode='sampling'
)
print(generated_story)
```

## Results

Training and validation loss/accuracy curves are plotted after training. Note that with a relatively small corpus and vocabulary, prediction accuracy is limited — generated text should be treated as an experimental/educational output rather than production-quality text generation.

## Limitations & Future Work

- Accuracy is constrained by the size of the training corpus; a larger or more diverse dataset would likely improve results.
- The current architecture could be extended with attention mechanisms or replaced with a Transformer-based model for better long-range coherence.
- Hyperparameters (embedding size, LSTM units, dropout rate) were tuned manually and could benefit from systematic search.

## License

Add your preferred license here (e.g., MIT).
