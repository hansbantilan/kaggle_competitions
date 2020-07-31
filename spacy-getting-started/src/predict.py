##########################################################################################
# Predictions
##########################################################################################
from KagglePipeline import * # KagglePipeline class
import spacy

# Instantiate an KagglePipeline object for the "Real or Not" Kaggle competition
realornot = KagglePipeline('train.csv')

# Print parameters of KagglePipeline instance
realornot.print_parameters()

# Load data into texts list and targets list
texts, targets = realornot.load_data()

# Make train/val/test split
train_texts, train_targets, remnant_texts, remnant_targets = realornot.split_data(texts, targets, split_ratio=0.9)
val_texts, val_targets, test_texts, test_targets = realornot.split_data(remnant_texts, remnant_targets, split_ratio=0.5)

# Convert targets into labels i.e. a dictionary keyed by 'cats' required by a spaCy TextCategorizer
train_labels = realornot.convert_to_cats(train_targets)
val_labels   = realornot.convert_to_cats(val_targets)
test_labels  = realornot.convert_to_cats(test_targets)

# Create spaCy model
nlp, _ = realornot.spacy_model(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)

# Evaluate model on the training set and on the validation set
train_accuracy = realornot.evaluate(nlp, train_texts, train_labels)
val_accuracy = realornot.evaluate(nlp, val_texts, val_labels)
print(f"Training set accuracy: {train_accuracy:.3f}")
print(f"Validation set accuracy: {val_accuracy:.3f}")

# Extract TextCategorizer from the nlp pipeline
textcat = nlp.get_pipe('textcat')

# Make predictions with model on test set and convert predicted_clas to 'real' or 'not'
predicted_class = realornot.predict(nlp, test_texts)
filename = 'example-by-example_predictions.txt'
f = open(filename, 'a')
for pred, text, label in zip(predicted_class, test_texts, test_labels):
    if label['cats'][textcat.labels[pred]]:
        f.write(f"{textcat.labels[pred]} (correct): {text} \n") 
    else:
        f.write(f"{textcat.labels[pred]} (incorrect): {text} \n")
print(f"\n---Example-by-example predictions on test set ... saved in {filename}--- \n\n")
