# Guitar-Chord-Classifier
### Pipeline for training a CNN to classify guitar chords from the Kaggle Guitar Chords V2 dataset.


The dataset can be found at: https://www.kaggle.com/datasets/fabianavinci/guitar-chords-v2

Dependencies are in the requirements.txt

Implementations of this project:
- Created csv files from the dataset folder
- Created a dataloader to preprocess and load the data
- During preprocessing sound samples get cut or padded to a standard size
- A plain CNN with four convolutional layers is used for the classification task
- The input is the mel spectrogram of a guitar chord sound and the output the class of the chord

Run by executing main.py
