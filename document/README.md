# question_classification

## BagOfWords Class Interface
This is a class for generating sentence bag-of-words vectors.
Interface supports two modes:
* glove
* random

### Glove
This mode loads the Glove Pre-trained word vectors: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download).
The Interaface uses the **100d** vectors.

How to initialise:
```python
bow = BagOfWords(embeddings='glove')
```

### Random
This mode builds a vocabulary of randomly initialised word vectors using the training data frequent words.

How to initialise:
```python
bow = BagOfWords(embeddings='random', training_data_path=PATH_TO_TRAIN_DATA_FILE, frequency_threshold=MIN_FREQUENCY)
```

### Class functions:
* tokenize(input_sentence):
    - input: string             e.g., "This is a sentence"
    - output: list of strings   e.g., ["This", "is", "a", "sentence"]
This function is used internally by the to_vector function, but it is public should you need to use it in a different context.

How to use:
```python
tokens = bow.tokenize('This is a sentence.')
```

* to_vector(input_sentence):
    - input: string                       e.g., "LABEL:label This is a sentence"
    - output: an 100 elemnt numpy array and a string representing the label
                                          e.g., [0.1124, 0.15151, -0.12516, ... 0.7872], "LABEL:label"

This is the main functionality of the class, these vectors represent the data model for the classifier.

How to use:
```python
sentence_vector = bow.to_vector('This is a sentence')
```