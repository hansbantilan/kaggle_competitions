##########################################################################################
# Learning curves
##########################################################################################
from KagglePipeline import * # KagglePipeline class
import matplotlib.pyplot as plt

# Instantiate an KagglePipeline object for the "Real or Not" Kaggle competition
realornot = KagglePipeline('train.csv')

# Load data into texts list and labels list
texts, labels = realornot.load_data()

# Make train/val/test split
train_texts, train_labels, remnant_texts, remnant_labels = realornot.split_data(texts, labels, split_ratio=0.9)
val_texts, val_labels, test_texts, test_labels = realornot.split_data(remnant_texts, remnant_labels, split_ratio=0.5)

# Convert labels into the dictionary format with the key 'cats' that a spaCy TextCategorizer requires
train_labels = realornot.convert_to_cats(train_labels)
val_labels   = realornot.convert_to_cats(val_labels)
test_labels  = realornot.convert_to_cats(test_labels)

# Define fractions of the training set to train on
fraction_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
fraction_accuracy_list = []
val_accuracy_list = []

for fraction in fraction_list:
    print(f"Training model with {fraction} of the training set")
    
    # Create spaCy models using varying fractions of the training set, and train for only one epoch each
    nlp, fraction_data = realornot.spacy_model(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, fraction=fraction, num_epochs = 1)
    
    # Evaluate model on a fraction of the training set and on the validation set
    fraction_accuracy = realornot.evaluate(nlp, *zip(*fraction_data))
    val_accuracy = realornot.evaluate(nlp, val_texts, val_labels)
    
    # Append to accuracy lists
    fraction_accuracy_list.append(fraction_accuracy)
    val_accuracy_list.append(val_accuracy)

# Plot (fraction_list, fraction_accuracy_list) and (fraction_list, val_accuracy_list)
filename = "learning_curves_" + str(fraction_list[0]) + "-" + str(fraction_list[-1]) + ".png"
plt.clf()
plt.plot(fraction_list,fraction_accuracy_list, color=(0,0,1,1), marker='o', label='on training set')
plt.plot(fraction_list,val_accuracy_list, color=(1,0,0,1), marker='o', label='on validation set')
plt.xlabel('Fraction of the training set used in training')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig(filename)

print(f"\n---Learning curves ... saved in {filename}--- \n\n")


