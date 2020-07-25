##########################################################################################
# Predictions
##########################################################################################
import utils
import spacy

def main():
	# Load data into texts list and labels list
	texts, labels = utils.load_data('train.csv')
	
	# Make train/val/test split
	train_texts, train_labels, remnant_texts, remnant_labels = utils.split_data(texts, labels, split_ratio=0.9)
	val_texts, val_labels, test_texts, test_labels = utils.split_data(remnant_texts, remnant_labels, split_ratio=0.5)
	
	# Convert labels into a dictionary keyed by 'cats' required by a spaCy TextCategorizer
	train_labels = utils.convert_to_cats(train_labels)
	val_labels   = utils.convert_to_cats(val_labels)
	test_labels  = utils.convert_to_cats(test_labels)

	# Create spaCy model
	nlp, _ = utils.spacy_model(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
	
	# Evaluate model on the training set and on the validation set
	train_accuracy = utils.evaluate(nlp, train_texts, train_labels)
	val_accuracy = utils.evaluate(nlp, val_texts, val_labels)
	print(f"Training set accuracy: {train_accuracy:.3f}")
	print(f"Validation set accuracy: {val_accuracy:.3f}")
	
	# Extract TextCategorizer from the nlp pipeline
	textcat = nlp.get_pipe('textcat')
	
	# Make predictions with model on test set and convert predicted_clas to 'real' or 'not'
	predicted_class = utils.predict(nlp, test_texts)
	f = open('predictions_on_test_set.txt', 'a')
	for pred, text, label in zip(predicted_class, test_texts, test_labels):
	    if label['cats'][textcat.labels[pred]]:
	        f.write(f"{textcat.labels[pred]} (correct): {text} \n") 
	    else:
	        f.write(f"{textcat.labels[pred]} (incorrect): {text} \n")
	print(f"\n---Example-by-example predictions ... saved in predictions_on_test_set.txt--- \n\n")

if __name__ == "__main__":
    main()
