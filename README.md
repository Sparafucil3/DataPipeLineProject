# Disaster Response Pipeline Project

This first run is a minimum viable product. Additional features can be added as necessary. It is intended to showcase the art of the possible. 

#### Table of Contents
[Headers](#Install)

<a name='Install'/>
### Install
This project uses the following libraries. All libraries are part of Anaconda: 

- sys
- pandas
- sqlalchemy
- re
- nltk
- sqlalchemy
- sklearn
- pickle
- flask
- ploty

All code in the project was created with Python 3.6.3

### Why this project
This project fulfills exercise 2 for the <a href='https://www.udacity.com/course/data-scientist-nanodegree--nd025'>Udacity Data Science</a> Nano Degree program. This program is based on data provided by <a href='https://appen.com/'>Figure Eight</a>. The data contains a corpus of tagged messages related to disasters I used to create a model categorizing those messages to create a web app for real-time categorization of new messages. The web app also provides some limitted visualiztions of the data. 

### Description
This project is executed in three segments: 
1. <a href='https://github.com/Sparafucil3/DataPipeLineProject/blob/master/data/process_data.py'>process_data.py<a>: This script expects two input files with a common "id" field, one containing messages and one containing categories. It ingests the files, merges them, cleans the data to prepare it for Machine Learning, and then stores it in a SQLite database table named "CleanedMessages".
2. <a href='https://github.com/Sparafucil3/DataPipeLineProject/blob/master/models/train_classifier.py'>train_classifier.py</a>: This script loads data from the SQLite DB created in step 1. It builds a Pipeline and a GridSearchCV to optimize finding the best RandomForrestClassifier possible. I picked a RandomForrestClassifier as it seems a best fit for this MVP. WARNING: the step "Training model" will take some time to execute. There is a lot going on in this step.
3. Lastly, this project contains a small, lightweight Flask app for deployment to a web-server. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
