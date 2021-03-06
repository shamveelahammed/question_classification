# Question Classification

## Tokenizer
This is a class to handle tokenization of sentences, as well as setting to lower-case, removing stopwords and punctuation.
The class interface takes as input parameter 'lower' which can be set to either True or False, to toggle on or off the setting to lowercase. 
### HOW TO USE
```python
from Tokenizer import Tokenizer

tokenizer = Tokenizer(lower=True) # set lower to either True or False if words need to be set to lowercase or not.
sentence = 'This acts as an example question to point out the capabilities, including stopwords ?'
print(tokenizer.tokenize(sentence))
```
```bash
['this', 'acts', 'example', 'question', 'point', 'capabilities', 'including', 'stopwords']
```

## WordEmbeddingLoader
This is a class that loads word embeddings, either GloVe or Random, by calling the load method. The class returns a dictionary
which maps each train word to an index, and a dictionary (the torch.nn.Embedding object) which maps each index to a tensor array.
By looking up the index of a word, we can extract its word embeddings using the embedding dictionary returned by the Loader.

For GloVe, this method has only one input parameter 'freeze' which can be True or False to toggle on and off the fine-tuning.
The method initializes the embedding as a torch.nn.Embedding object from the glove.txt file in data/. This object also takes 'freeze' as an input parameters.

For Random, this method takes as input parameters 'random=True', 'data_path=LOCATION_OF_TRAIN_DATA' and 'frequency_threshold=K'.
If random flag is set to true, the method will expect the location of the training data, to load the words from it, and assing
each an random embedding. The frequency threshold K value defines how frequent a word needs to be found in the training data
to be loaded in the embedding dictionary and given a random embedding.
### HOW TO USE
* GloVe
```python
from torch import LongTensor
from WordEmbeddingLoader import WordEmbeddingLoader

# freeze is by default False, so only needs specifying when true.
word_to_index, embedding = WordEmbeddingLoader.load(freeze=True) 
index =  LongTensor([word_to_index['name']])
print(embedding(index))
```
```bash
tensor([[ 1.1326e-01, -2.3055e-01,  4.6840e-01, -2.6068e-01,  1.2871e-01,
          3.8373e-01, -3.2314e-02, -5.7986e-01,  1.8424e-01,  4.6796e-02,
         -5.5893e-01,  2.7794e-01,  7.4838e-01,  3.3575e-01,  2.6834e-02,
         -4.4505e-01,  1.4755e+00, -1.4081e-01, -3.1658e-01,  5.7686e-01,
         -1.8440e-01, -1.0370e-01, -2.2519e-01,  3.4614e-01, -2.1930e-01,
         -2.3868e-01, -1.2845e-01, -7.7772e-01,  1.6957e-01,  2.9432e-02,
          4.7705e-01,  8.5881e-01, -5.3617e-02,  1.0770e-01,  9.6440e-02,
          2.7325e-01,  4.4933e-02,  1.3310e-03,  5.0395e-02, -4.8147e-01,
          1.7561e-01,  1.9419e-01,  5.1495e-01, -5.6149e-01,  1.1211e-01,
          2.5591e-01,  2.9011e-02, -3.4461e-03, -1.8478e-01, -2.0650e-01,
          1.0358e-01,  7.5532e-01,  9.4688e-01,  8.4278e-01, -7.0051e-01,
         -2.4433e+00, -9.1913e-01, -1.0451e-01,  9.0954e-01,  2.5491e-01,
          2.1129e-01,  1.2046e+00, -9.1615e-02,  3.0364e-01,  1.3768e+00,
         -5.8152e-01,  3.8085e-01,  1.5504e-01,  2.0049e-01,  7.3361e-04,
         -6.2300e-01,  2.0244e-01, -3.0387e-01, -8.1412e-01,  3.8805e-01,
          2.1271e-01, -4.1525e-02, -4.5596e-02, -1.1508e+00, -2.5826e-01,
         -8.9721e-02, -1.1256e+00, -2.5124e-01, -2.8010e-01, -1.0334e+00,
         -1.6813e-01, -4.0975e-01, -1.0685e+00,  7.4311e-01,  8.3244e-02,
         -3.3616e-01, -8.8150e-02,  1.5401e-01,  4.7736e-01, -1.8272e-01,
         -2.5543e-01, -8.9365e-01, -4.6822e-01,  1.9834e-01, -4.8772e-02]],
       grad_fn=<EmbeddingBackward>)
```
* Random
```python
from torch import LongTensor
from WordEmbeddingLoader import WordEmbeddingLoader

# random is by default False - which points to GloVe.
# data_path and frequency_threshold are by default None, as they are not needed for Glove.
word_to_index, embedding = WordEmbeddingLoader.load(random=True, data_path='data/train.txt', frequency_threshold=2)
index = LongTensor([word_to_index['name']])
print(embedding(index))
```
```bash
tensor([[ 3.9767, -3.7362,  1.3508,  5.3946,  6.1296,  7.9175,  4.6109,  9.7723,
          5.1597, -4.6943,  0.4804, -0.8930, -4.8745,  4.9583,  4.7207, -3.5724,
         -4.5541,  8.2829, -4.4200,  2.3837,  7.1392, -7.8685, -9.2198, -3.9581,
         -5.2689,  6.4879, -2.3979,  4.8083, -6.2414,  5.9522,  4.7746, -3.8616,
         -3.2186, -6.1294,  8.4027,  3.4032,  6.0694, -5.7826, -2.9766, -8.0869,
          2.0886,  8.3767,  4.2533, -0.4216,  1.9670, -8.5290, -3.4193, -5.1419,
         -5.9387, -4.3393,  5.2120,  4.2373, -3.9011, -1.4666,  5.8140, -4.5459,
         -7.6593, -0.2991,  8.4109, -1.9186, -4.0094,  6.0459, -5.5141, -1.6606,
         -1.4810,  9.5261, -0.5930,  3.5788, -0.8000, -3.0614,  2.9651,  5.5222,
         -2.4042, -1.3364, -2.7648,  9.9728,  4.3789, -2.1103, -2.0517, -0.1833,
          2.0719, -3.1465,  3.9070, -3.6225, -6.6028, -6.7624, -4.3794,  3.1722,
          0.4343,  9.8298, -4.8722,  9.1777, -8.6320, -3.1353,  6.2808, -9.7483,
         -7.1193, -6.6784, -4.9679,  5.2280]], grad_fn=<EmbeddingBackward>)
```
## BagOfWords
This is an class, that, initialized with the word_to_index and embedding dictionary returned by WordEmbeddingLoader.load method,
returns via the get_vector method a vector representation of the sentence as a Bag Of Words. The BagOfWords also returns the label
of the data, if this exists.
### HOW TO USE
```python
from WordEmbeddingLoader import WordEmbeddingLoader
from BagOfWords import BagOfWords
word_to_index, embedding = WordEmbeddingLoader.load()
bow = BagOfWords(embedding, word_to_index)

vector, label = bow.get_vector("ENTY:food What do you get by adding Lactobacillus bulgaricus to milk ?")
print(vector)
print(label)
```
```bash
tensor([[-1.8807e-01,  2.7060e-01,  1.4559e-01, -3.8446e-01, -4.4511e-01,
          1.6268e-01, -4.6611e-02,  1.6292e-01,  1.8894e-01, -2.2770e-01,
          9.3888e-03,  4.3767e-02,  1.5394e-01,  6.6573e-02, -5.0421e-02,
         -2.3298e-01,  1.4011e-02,  1.3735e-01, -3.9502e-01,  4.2976e-01,
          2.4106e-01,  6.1968e-03, -5.0587e-02, -1.6007e-01, -1.2037e-02,
          2.8739e-01, -2.9323e-02, -4.5859e-01,  1.8303e-01, -2.0574e-02,
          4.8003e-02,  4.9990e-01, -3.2117e-02,  8.8852e-03,  2.2014e-02,
          3.4009e-01,  9.5243e-02,  3.8049e-02, -1.7309e-02, -2.9080e-01,
         -2.8072e-01, -1.7928e-01, -2.7879e-01, -4.2960e-01, -1.6043e-01,
          1.6936e-01, -1.3036e-01, -3.7004e-01, -2.2629e-01, -7.5640e-01,
         -2.2675e-02,  1.6486e-01, -9.0330e-02,  8.4934e-01, -2.3796e-01,
         -1.7485e+00, -1.0571e-02, -5.9525e-02,  1.1452e+00,  2.1251e-01,
         -8.0536e-02,  5.9981e-01, -3.0237e-01,  5.3585e-03,  7.0007e-01,
          2.6743e-01,  5.6389e-01,  3.0480e-01,  1.1357e-01, -4.5585e-01,
         -3.7189e-02, -1.3849e-01, -1.6728e-03, -2.4724e-01,  1.0204e-01,
          3.2366e-01, -1.5010e-01, -1.3200e-01, -5.3180e-01,  4.6464e-02,
          4.1398e-01, -1.3672e-01, -5.7774e-01,  5.6923e-02, -1.0517e+00,
         -8.7332e-02,  1.1316e-01, -3.8766e-02, -3.6268e-01, -2.6791e-01,
         -2.8019e-02, -2.1037e-01, -2.4737e-01, -3.1128e-01, -3.0808e-01,
         -2.6708e-01, -1.3435e-01, -2.9563e-01,  1.7983e-01,  2.3759e-01]],
       grad_fn=<DivBackward0>)
ENTY:food
```

## BiLSTMInterface
This is an interface class, that initialized with the location of the training data, and the lowercase flag needed by the Tokenizer, load the data and prepares the word_to_index and the label_to_index dictionaries for BiLSMT to use.

The class method load_and_train_bilstm, loads the word embeddings, either random or pretrained (specified by usePretrained param),
and initializes a BiLSTM object (torch.nn.Model calling torch.nn.LSTM) and trains it using the word_to_index, label_to_index, and
the word embeddings. The freeze parameters of this method is passed along to the glove loader, to either freeze or fine-tune. This method requires the embeddings dimension and the hidden dimension, to pass them along to the BiLSTM object.

The class method get_vector has the same interface as the BagOfWords class, which takes a sentence, and returns the label and a vector which is resulted by passing the tokenized sentence to the BiLSTM. This vector has 50 elements - which reflect the number of classes.

### HOW TO USE
```python
from BiLSTMInterface import BiLSTMInterface

EMBEDDING_DIM = 32
HIDDEN_DIM = 32
bilstm = BiLSTMInterface('../data/train_label.txt', lowercase=True)
bilstm.load_and_train_bilstm(EMBEDDING_DIM, HIDDEN_DIM, freeze=False, usePretrained=False)

label, vector = bilstm.get_vector("ENTY:food What do you get by adding Lactobacillus bulgaricus to milk ?")
print(vector)
print(label)
```
```bash
tensor([[-1.8807e-01,  2.7060e-01,  1.4559e-01, -3.8446e-01, -4.4511e-01,
          1.6268e-01, -4.6611e-02,  1.6292e-01,  1.8894e-01, -2.2770e-01,
          9.3888e-03,  4.3767e-02,  1.5394e-01,  6.6573e-02, -5.0421e-02,
         -2.3298e-01,  1.4011e-02,  1.3735e-01, -3.9502e-01,  4.2976e-01,
          2.4106e-01,  6.1968e-03, -5.0587e-02, -1.6007e-01, -1.2037e-02,
          2.8739e-01, -2.9323e-02, -4.5859e-01,  1.8303e-01, -2.0574e-02,
          4.8003e-02,  4.9990e-01, -3.2117e-02,  8.8852e-03,  2.2014e-02,
          3.4009e-01,  9.5243e-02,  3.8049e-02, -1.7309e-02, -2.9080e-01,
         -2.8072e-01, -1.7928e-01, -2.7879e-01, -4.2960e-01, -1.6043e-01,
          1.6936e-01, -1.3036e-01, -3.7004e-01, -2.2629e-01, -7.5640e-01]],
       grad_fn=<DivBackward0>)
ENTY:food
```
## FeedForwardNetwork
This is the Feed-forward Neural Network classifier class. It trains Bag of Words or BiLSTM models based on user's settings specified in the config file.

The constructor takes 4 arguments; hidden layer sizes, embedding parameters, max epoch, and learning rate. The embedding parameters specify settings such as either to use BOW or BiLSTM (method), the data path, either to use random initialisation or GloVe (random), lowercase, freeze, and so on. These settings can be modified by the user in the parameter.config file.

Before training, we need to obtain validation data from the dev.txt file. These data is then passed as arguments to the .fit() method. In each epoch, the model will be validated with these validation data, and the loss is computed. By the end of the training, the best model with the lowest validation loss is returned.
### HOW TO USE
```python
from torch import LongTensor
from FeedForwardNetwork import Feedforward

# parameter initialisations
embedding_params = {
       "method":    config['method'],
       "data_path": config['train_file'],
       "random":  (config['init_method'] == 'random'),
       "frequency_threshold": config['bow_frequency_threshold'],
       "vector_size": config['weights_vector_size'],
       "lowercase": config['lowercase'],
       "freeze": config['freeze'],
       "training_size": config['training_size'],
       "temp_train": config['temp_train']
}
maxEpoch = config['maxepoch']
learningRate = config['learningRate']

# there are 3 layers, hence list must have 3 values
hidden_layer_sizes = config['hidden_layer_sizes']

# create NN Classifier Instance
# takes 3 arguments: hidden layer sizes, embedding params, epoch, and learning rate
model = Feedforward(hidden_layer_sizes,
                     embedding_params, maxEpoch, learningRate)
# get validation data
x_val, y_val = get_validation_data(config, model)
# Training the model
# return model with best validation loss
model = model.fit(x_val, y_val)

def get_validation_data(config, model):
    """
    TODO: This is just a stub - complete with relevant calls for processing (word embeddings, bow/bilstm) and testing against validation
    """

    # use model's sentence model - BOW or BiLSTM
    sentence_model = model.sentence_model

    # Get Text embedding for testing
    x_val, y_val_arr = model._get_text_embedding(
        sentence_model, config['dev_file'])

    dic = model.full_class_dictionary

    # Convert arrays to Tensors
    y_val = torch.from_numpy(
        np.array([dic[v] for v in y_val_arr])).long()

    return x_val, y_val
```

## Evaluator
This class handles the evaluation predicted labels against the actual labels of train, dev and test data. It calculates Precision, Recall, and micro-averaged F1 score. It takes two arguments as constructor, first the squeezed predicted tensor, and the second is the actual labels, in the form of integer indices 0 - 49. This class converts the former into integer form, and can be accessed through the predicted_labels class variable.
### HOW TO USE
```python
from Evaluator import Evaluator

# Evaluator
evaluator = Evaluator(y_pred.squeeze(), y_test)
# getPrecision return True Positive count and Precision score
correct_count, precision = evaluator.get_Precision()
f1 = evaluator.get_f1_score()

# you can also access the predicted and actual labels, both in integer index form like the following:
print(evaluator.predicted_labels)
print(evaluator.actual_labels)
```

## ConfusionMatrix
This class handles the generation of confusion matrix from pairs of predicted and actual labels. The constructor takes three arguments; the class dictionary, predicted labels (integer index form) and actual labels (integer index form). The getConfusionMatrix method returns a Pandas data frame object, which can later be passed to the Seaborn heat map generator. This class works seamlessly with the Evaluator class.
### HOW TO USE
```python
from ConfusionMatrix import ConfusionMatrix

conMatGenerator = ConfusionMatrix(model.class_dictionary, evaluator.predicted_labels, evaluator.actual_labels)
cm_df = conMatGenerator.getConfusionMatrix()
print(cm_df)

# the data frame object is passed to Seaborn's heatmap method to produce heat map
heat_map = sns.heatmap(cm_df, center=0,  vmin=0, vmax=15)
plt.show()
```

## buildOutputTxt
This method is used to generate an output.txt file, each line is the test questions, and the second and third columns are its predicted and actual labels respectively. THIS FILE NEEDS TO BE VIEWED FULL SCREEN.
```python
def buildOutputTxt(dataFile, y_arr, y_pred, dic, score):
    # convert classes from indices back to string form
    y_pred_list = []
    for item in y_pred:
        for key, value in dic.items():
            if value == item:
                y_pred_list.append(key)

    # Open Test Data File
    with open(dataFile, encoding="ISO-8859-1") as f:  # 515
        testData = f.read().split('\n')
    f.close()

    # get only the question
    questions = []
    for sentence in testData:
        question_words = sentence.split(' ')[1:]
        question_sentence = ' '.join(question_words)
        if len(question_words) != 0:
            questions.append(question_sentence)

    # create padded alignment
    questions_padded = padding(questions)
    y_pred_padded = padding(y_pred_list)

    with open('output.txt', 'w') as f:
        f.write('THIS FILE MUST BE VIEWED FULL SCREEN\n')
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        f.write(
            '\t\t\tQuestions\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tPredicted\t\tActual\n')
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        for i in range(0, len(questions_padded)):
            f.write('{}{}{}\n'.format(
                questions_padded[i], y_pred_padded[i], y_arr[i]))
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
        f.write(
            '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tF1 Accuracy (%): {}\t\t\n'.format(score * 100))
        f.write(
            '----------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    f.close()
```
```bash
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
			Questions																												Predicted		Actual
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
What Southern California town is named after a character made famous by Edgar Rice Burroughs ?                                        HUM:ind       LOC:city
Where is Rider College located ?                                                                                                      LOC:other     LOC:other
In what war was the first submarine used ?                                                                                            NUM:date      ENTY:event
What country has the port of Haifa ?                                                                                                  LOC:country   LOC:country
What was the name of the Protestant revolt against the supremacy of the Pope ?                                                        HUM:ind       ENTY:event
What character in The Beverly Hillbillies has the given names Daisy Moses ?                                                           HUM:ind       HUM:ind
Which drug is commonly used to treat AIDS ?                                                                                           ENTY:dismed   ENTY:dismed
Whose first presidential order was : `` Let 's get this goddamn thing airborne '' ?                                                   ENTY:other    HUM:ind
What is another name for nearsightedness ?                                                                                            ENTY:termeq   ENTY:termeq
What Scandinavian country covers 173 , 732 square miles ?                                                                             LOC:country   LOC:country
What is Stefan Edberg 's native country ?                                                                                             LOC:country   LOC:country
What do you need to do to marry someone in jail ?                                                                                     DESC:reason   DESC:desc
What was the Great Britain population from 1699-172 ?                                                                                 NUM:count     NUM:count
What is the length of border between the Ukraine and Russia ?                                                                         DESC:def      NUM:dist
Who was The Pride of the Yankees ?                                                                                                    HUM:ind       HUM:ind
```

## Training Classifier
### HOW TO USE
- Ensure all libraries are installed
```bash
pip3 install torch
pip3 install -U numpy
pip3 install -U pandas
pip3 intall -U seaborn
pip3 install -U PyYAML
```

- To train the model
```bash
python question_classifier.py train --config_file parameter.config 
```
```bash --config_file  ``` the hyper-paramters to train the model and NN <br />

- To test the model
```bash
python question_classifier.py test --config_file parameter.config 
```
```bash --config_file  ``` the hyper-paramters to test the model and NN <br />

