## Music Mood Classification Using the Million Song Dataset


### Technical Summary 

If you want to see a quick summary of the methods & results, here are the [slides](http://bit.do/joydiv). 

A detailed technical report is available as a [PDF](https://github.com/bhavika/JoyDivision/blob/master/report/report.pdf)


### Installation

1. Clone/download ZIP from https://github.com/bhavika/JoyDivision.git
2. cd JoyDivision
3. If you want to create a virtual environment, run virtualenv <the name of the environment, say 'tekwani'>
4. To begin using the virtual environment, you must activate it.
   $ source tekwani/bin/activate
5. Now install the packages specified in requirements.txt. You can do this using
   pip freeze > requirements.txt (freeze the current state of the environment)
   pip install -r requirements.txt


### Running the solution

1. Download the data from the Google Drive link here: http://bit.do/datasets
   The total download size should be about 2.8 GB. 
2. Move the downloaded files to tekwani/data and check that it contains the following files:

    -fullset.pkl
    -train.pkl
    -test.pkl

The folder JoyDivision/explore contains plots, output for different estimators' grid search results, scripts to handle h5 files and a list of getter methods 
for h5 files (hdf5_getters.txt)

The file `evaluation.py` is the final file that generates results as shown in the report. 
It will run 6 models for 3 feature combinations - audio and descriptive, descriptive only, audio only.


### Other files

1. `create_dataset.py` builds the dataset and randomly splits it into a 60-40 distribution fof train and test sets.
2. `models.py` evaluates feature importance for ensemble estimators and performs cross validation for all estimators.
3. `read_h5.py` is used to pull data out of HDF5 files. To run this, you need to download the Million Song Subset and place it in `data`
4. `spotify.py` searches for Track IDs for the songs we have labeled in Spotify's database.
5. `spotify_audio_features.py` fetches Danceability, Energy, Speechiness, Acousticness, Valence and Instrumentalness for all the track IDs we were able to get from 
    `spotify.py`. 
6. `rfe.py` gets the ranking of features. I use the number of optimal features (n) obtained here to select the top n features in `feature_importance.py` for the estimators.
7. `feature_combinations.py` evaluates the importance of features when compared to the groups they're combined with. For e.g., Descriptive features paired with timbre, 
    audio features paired with descriptive, etc
8. `get_train_test.py` serves the train and test sets (*.pkl) to any file that imports it.
7. `scratch\labels.csv` contains the full list of songs and the labels I assigned to them. 
8. `scratch\models.out` contains the output for `models.py`. These are only cross validation results. 
9. `learning_curve.py` plots the training score and cross validation score for an SVM with a linear kernel.
10. `*.out` files - output files. 
11. `hdf5_getters.py` is an interface provided along with the Million Song Dataset by LabROSA (Columbia University). It is used to read HDF5 files which is the initial
    form of the Million Song Dataset.
12. All files named `gridsearch_.py` are used to do a hyperparameter search for the models used. These models are ADABoostClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, SVM, KNearestNeighbour and RandomForestClassifier.

