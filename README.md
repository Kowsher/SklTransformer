# SklTransformer: Sci-Kit Learn Transformer


## Sci-Kit Learn Transformer
SklTransformer or SklT is a sentence or document embedding based on transformers, supervised fine-tuning/transfer learning, and showed the state of the art result for the Sklearn classifier(Logistic Regression, SVM, Naive Bayes, KNN, Decision Tree). An extra exponential function has been added to the final pooling output layer of bert and it gives state-of-the-art results than the previous models proposed. The architecture of Google BERT shows the [CLS] layer uses a softmax function on the pooling layer to predict the output and Huggingface uses a tanh function. Both of these methods have complications because values can be negative or too small. Adding an exponential layer converts values of the pooling layer into a positive range from 0.3 to 2.7. Consequently, every vector get a positive and non-zero unit which helps to keep significance of every single vector in every mathematical operation. Besides it extent the vector distance, which is good for decision making. SklTransformer has been applied to fine-tune the features for further classification using machine learning.



## Installation
To install the latest release, we can do :

``` python
!pip install SklTransformer
```


## Getting started


In order to apply SklTransformer, as described here, `SklTransformer` function like this:
``` python
import SklTransformer
```

## With fine tuning
It's a good idea to fine tuning the BERT model according to your dataset before getting sentnence embedding for the use of sklearn classifier.
If we have already model then we can simply read and load the model as :

``` python
import SklTransformer

sklt =SklTransformer.SklT(tokenizer_name="bert-base-uncased", model_name="bert-base-uncased", fine_tuning=True, X_train=X_train, y_train=y_train,X_val=X_test, y_val=y_test, nub_epoch=10,save_steps=500, save_path = '/content/drive/MyDrive/spam/')
```
'sklt' will carray a object of SklTransformer
In order to fine-tuning, we have to check the parameter 'fine-tuning' as true. By default it's false.

-> For supervised fine tuning we have to check some parameters 

-> tokenizer_name (Need to pass model name or path),

-> model_name (Need to pass tokenizer_nameor path),

-> fine_tuning=False (Need to pass as true),

-> X_train (Need to pass traing text, it can be array, list),

-> y_train (Need to pass traing labels, it can be array, list),

-> X_val (Need to pass validation text, it can be array, list),

-> y_val (Need to pass validation labels, it can be array, list,
nub_epoch (Number of epochs, by default 10),

-> batch_size (Number of batch sizes, by default 8),

-> save_steps (Number of saving checkpoint and evaluation, by default 5000),

-> save_total_limit (Number of saving checkpoint, by default 1 that means, it will only save the best checkpoint)

-> stopping_callback (Early stopping callback of traing, by default 4)

-> save_path (Model saving path),

-> max_length (Maximum length of every sample, by default 512)

-> device (In which device, we want to traing model, default automatic choose of device according to environment)

## Playing with Device selection
Device selection is a very important step in SklTransformer. In general, the training device can be automatically selected 

-> If we set up our machine with TPU, it will select as XLA

-> If we set up our machine with GPU, it will select as CUDA

-> If we set up our machine with CPU, it will select as CPU

In general, we want to pass particularly suitable device we can pass it as parameter such as xla, cuda, cpu

In order to training with TPU, it may be required 'torch_xla-1.9'. In this case, before importing SklTransformer, we need to install torch_xla-1.9

``` python
!pip install https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
```

## Without fine tuning

If we have a fine tuned model or we do not want to fine tuning then we can just go with simple way (It's recommodeation for supervised fine-tuning of lanague model for better result)

``` python
import SklTransformer

sklt =SklTransformer.SklT(tokenizer_name="bert-base-uncased", model_name="bert-base-uncased")
```
## Playing with sentence or document embedding
The primary purpose of SklTransformer is the fixed-length (768) of single dimension of word embedding for every sentence or document so that the masked language model such as bert can be used as the embedding of sklearn machine learning classifiers.

if 'sklt' is the object of SklTransformer after fine-tuned 

``` python
X_train = sklt.fit_transform(X_train)
X_test = sklt.transform(X_test)
......
```
It will return as numpy array which is suitable for any sklearn classifier

## For better uses
For better use, it will be good if we remove TensorFlow. In very few cases TensorFlow can make issues. But in our experiment, we did not get any issues yet. 
To uninstall TensorFlow we can just write a simple code
``` python
!pip uninstall -y tensorflow
```
In the training time of the TPU machine, it's better to use torch_xla-1.9 which has been described above
But when a model is running on CPU, it is highly suggested to remove torch_xla-1.9 
To uninstall torch_xla-1.9

``` python
!pip uninstall -y https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
```
## Development
install_requires=[
		'transformers==4.8.2',
		'torch==1.9.0',
		'tqdm==4.41.1',
		'numpy==1.19.5',
		'sklearn',],
