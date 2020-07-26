##########################################################################################
# KagglePipeline class
##########################################################################################
import pandas as pd
import spacy
from spacy.util import minibatch
import random

class KagglePipeline:

	def __init__(self, csv_file, fraction=1.0, num_epochs = 10, minibatch_size = 8, architecture='simple cnn'):
		self._csv_file = csv_file
		self._fraction = fraction
		self._num_epochs = num_epochs
		self._minibatch_size = minibatch_size
		self._architecture = architecture

	def print_parameters(self):
		print(f"----------------------------------------------------")
		print(f"Load data from {self._csv_file}")
		print(f"Train with a {self._fraction} fraction of the training set,")
		print(f"For {self._num_epochs} epochs, using a minibatch size of {self._minibatch_size},")
		print(f"And a {self._architecture} architecture")
		print(f"----------------------------------------------------")

	def load_data(self):
	    """Returns texts and targets for each (text,target) example in shuffled order from csv_file
	    
	        Arguments
	        ---------
	        csv_file: string of csv file name with columns 'id', 'keyword', 'location', text', 'target'
	    """
	    data = pd.read_csv(self._csv_file)
	
	    # Shuffle data
	    shuffled_data = data.sample(frac=1, random_state=1)
	
	    # Define texts from 'text' column and labels from 'target' column (HB: remember to use 'keyword','location' columns too)
	    texts = shuffled_data['text'].values
	    targets = shuffled_data['target'].values
	    
	    return texts, targets
	
	def split_data(self, texts, targets, split_ratio=0.9):
	    """Returns set1 texts, set1 targets, set2 texts, set2 targets for each (texts,target) example
	    
	        Arguments
	        ---------
	        texts: list of texts
	        targets: list of targets
	        split_ratio: float {# of set1 examples} / {# of set2 examples} 
	    """
	    # Convert split ratio to a split index
	    split_idx = int(len(texts) * split_ratio)
	    
	    # Split set1 and set2 texts
	    set1_texts = texts[:split_idx]
	    set2_texts = texts[split_idx:]
	
	    # Split set1 and set2 targets
	    set1_targets = targets[:split_idx]
	    set2_targets = targets[split_idx:]
	    
	    return set1_texts, set1_targets, set2_texts, set2_targets
	
	def convert_to_cats(self, targets):
	    """Returns a dictionary keyed by 'cats' required by a spaCy TextCategorizer, 
		   in the format {'cats': {"real": bool(target), "not": not bool(target)}}
	    
	        Arguments
	        ---------
	        original_labels: list of labels
	    """
	    labels = [{'cats': {"real": bool(target), "not": not bool(target)}} for target in targets]
	    
	    return labels
	
	def train(self, model, train_data, optimizer, minibatch_size=8):
	    losses = {}
	    random.seed(1)
	    random.shuffle(train_data) # re-shuffle the training data each time this method is called
	
	    # Create the batch generator of minibatches with minibatch_size
	    batches = minibatch(train_data, size=minibatch_size)
	    # Iterate through minibatches
	    for batch in batches:
	        # Each batch is a list of (text, label) but we need to
	        # send separate lists for texts and labels to update().
	        # This is a quick way to split a list of tuples into lists
	        texts, labels = zip(*batch)
	        model.update(texts, labels, sgd=optimizer, losses=losses)
	
	    return losses # return loss as a dictionary {model’s name : accumulated loss}
	
	def predict(self, model, texts):
	    # Use the model's tokenizer to tokenize each input text
	    docs = [model.tokenizer(text) for text in texts]
	
	    # Use textcat to get the scores for each doc
	    textcat = model.get_pipe('textcat')
	    scores, _ = textcat.predict(docs)
	
	    # From the scores, find the class with the highest score/probability
	    predicted_class = scores.argmax(axis=1)
	
	    return predicted_class
	
	def evaluate(self, model, texts, labels):
	    """ Returns the accuracy of a TextCategorizer model.
	
	        Arguments
	        ---------
	        model: spaCy model with a TextCategorizer
	        texts: list of texts
	        labels: list of labels
	
	    """
	    # Extract TextCategorizer from the nlp pipeline
	    textcat = model.get_pipe('textcat')
	    
	    # Get predictions from textcat model 
	    predicted_class = self.predict(model, texts)
	
	    # Create a boolean list indicating correct predictions
	    correct_predictions = [label['cats'][textcat.labels[pred]] for pred, label in zip(predicted_class, labels)]
	
	    # Calculate the accuracy i.e. the number of correct predictions divided by all predictions
	    accuracy = sum(correct_predictions)/len(predicted_class)
	
	    return accuracy
	
	def spacy_model(self, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, fraction=1.0, num_epochs = 10, minibatch_size = 8, architecture='simple cnn'):
	    """Returns spaCy model, 
	       and a list of (train_texts, train_labels) from the fraction of training data used in training
	    
	        Arguments
	        ---------
	        train_texts: list of texts from training set
	        train_labels: list of labels from training set
	        val_texts: list of texts from validation set
	        val_labels: list of labels from validation set
	        test_texts: list of texts from test set
	        test_labels: list of labels from test set
	        fraction: float fraction of training set to be used to train model
	        num_epochs: integer number of epochs of the optimization loop
	        minibatch_size: integer size of a minibatch
	        architecture: string architecture choice
	                      'ensemble' i.e. ensembled bag-of-words + CNN with mean pooling & attention
	                      'simple cnn' i.e. vectors are mean pooled and used as features in a CNN
	                      'bow' i.e. bag-of-words
	    """
	    # Create blank Language class
	    nlp = spacy.blank("en")
	
	    # Create a TextCategorizer with exclusive classes
	    textcat = nlp.create_pipe(
	                  "textcat",
	                  config={
	                    "exclusive_classes": True,
	                    "architecture": architecture})
	
	    # Add the TextCategorizer to the nlp pipeline
	    nlp.add_pipe(textcat)
	
	    # Add labels to text classifier
	    textcat.add_label("real")
	    textcat.add_label("not")
	
	    # Fix seed for reproducibility
	    spacy.util.fix_random_seed(1)
	    random.seed(1)
	
	    # Create training data
	    train_data = list(zip(train_texts, train_labels))
	    
	    # Convert fraction to a fraction index
	    fraction_idx = int(len(train_data) * fraction)
	    
	    # Keep only a fraction of training data
	    fraction_data = train_data[:fraction_idx]
	    
	    # Train and on-the-fly evaluate on the training set in a loop, num_epochs times
	    optimizer = nlp.begin_training()
	    for i in range(num_epochs):
	        losses = self.train(nlp, fraction_data, optimizer, minibatch_size=minibatch_size)
	        fraction_accuracy = self.evaluate(nlp, *zip(*fraction_data))
	        print(f"Loss after epoch {i}: {losses['textcat']:.3f} \t Accuracy ({fraction} of training set): {fraction_accuracy:.3f}")
	    
	    # Print success
	    print(f"---------------------------------------------")
	    print(f"spaCy model trained... saving")
	    print(f"---------------------------------------------")
	        
	    return nlp, fraction_data
