# Disaster Response Pipeline Project

- [Disaster Response Pipeline Project](#disaster-response-pipeline-project)
    + [Install](#install)
    + [Why this project](#why-this-project)
    + [Description](#description)
    + [Instructions](#instructions-)
    + [Data analysis and issues](#data-analysis-and-issues)
    + [Model Performance](#model-performance)
    + [Future Opportunities](#future-opportunities)

### Install 
This project uses the following libraries. All libraries are part of <a href='https://anaconda.org/'>Anaconda</a>: 
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
This project fulfills exercise 2 for the <a href='https://www.udacity.com/course/data-scientist-nanodegree--nd025'>Udacity Data Science</a> Nano Degree program. This program is based on data provided by <a href='https://appen.com/'>Figure Eight</a>. The data contains a corpus of tagged messages related to disasters I used to create a model categorizing those messages to create a web app for real-time categorization of new messages. The web app also provides some limitted visualiztions of the data. This first run is a minimum viable product (MVP). Additional features can be added as necessary. It is intended to showcase the art of the possible.

### Description 
This project is executed in three segments: 

1. <a href='https://github.com/Sparafucil3/DataPipeLineProject/blob/master/data/process_data.py'>process_data.py</a>: This script expects two input files with a common "id" field, one containing messages and one containing categories. It ingests the files, merges them, cleans the data to prepare it for Machine Learning, and then stores it in a SQLite database table named "CleanedMessages".
2. <a href='https://github.com/Sparafucil3/DataPipeLineProject/blob/master/models/train_classifier.py'>train_classifier.py</a>: This script loads data from the SQLite DB created in step 1. It builds a Pipeline and a GridSearchCV to optimize finding the best RandomForrestClassifier possible. I picked a RandomForrestClassifier as it seems a best fit for this MVP. WARNING: the step "Training model" will take some time to execute. There is a lot going on in this step.
3. Lastly, this project contains a small, lightweight Flask app for deployment to a web-server.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/

### Data analysis and issues
* This data has significant imbalances in the tag types. These imbalances makes it very difficult to pull relevant discriminatory attributes to identify those under-represented message types. It is very difficult for machine learning--which relies on examining many examples--to create the necessary selection criteria to identify these classes. If possible, it would be best to add additional observations to the initial dataset to help build a proper classifier. As it is not possible to go back to the data provider for additional examples we are limited to the data we have on hand. To compensate for this, I tried the following: 
    - **class_weight:** This <a href='https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html'>RandomForrestClassifier</a> class_weight hyper-parameter has an option called **'balanced'** which weights the value of examples inversely proportional to class frequencies in the input data as defined by the equation n_samples / (n_classes X np.bincount(y)). In our specific case, the highest scoring model used class_weight=None meaning no effort was made in this MVP model to account for imbalances in the class. 

### Model Performance
The class-by-class scoring is represented in the table below. The "occured" column lists the number of times that tag occurred in the dataset provided. In those instances where there are no examples, we would expect no score.
- **precision:** represents the accuracy of true predictions for that category type, AKA the TRUE POSITIVE RATE. For instance, with 'water", when we predict the message is related to water, we are correct 88% of the time.
- **recall:** the ability of the model to identify all the positive sameples in the Test data. For instance, with water, we know there are 350 samples in the TEST corpus and we only identify 12% of them. This is sometimes called "sensitivity". 
- **F1-score:** represents a balance between precision and recall. The higher the score, the more accurate & sensitive the model is to that label. 

True Postive + False Positive represents the workload of messages results which must be processed. False Negative represents the risk of missing something that is actually important. Determining the viability of this model is dependent upon the business case. If it is imperative that istances are not missed, recall will becomes the most important metric but will likely increase the false positive rate for that label and precision will suffer. This might be acceptable if the cost for processing false positives is relatively low. If resources are limited or the cost of handling positive results is high, accuracy would be be a better measure as this reduces the number of false positives, but likely also reduces the model's sensitivity and also increases the risk of something important being missed. 

|           label         | precision | recall | f1-score | occured  |
| :---------------------- | :-----: | :------: | :------: | :------: |
|             related     |  0.84   |   0.93   |   0.88   |   3977   |
|              request    |  0.84   |   0.44   |   0.58   |    869   |
|                offer    |  0.00   |   0.00   |   0.00   |     18   |
|          aid_related    |  0.74   |   0.55   |   0.63   |   2090   |
|         medical_help    |  0.59   |   0.07   |   0.12   |    393   |
|     medical_products    |  0.67   |   0.07   |   0.13   |    260   |
|    search_and_rescue    |  1.00   |   0.01   |   0.01   |    146   |
|             security    |  0.00   |   0.00   |   0.00   |     97   |
|             military    |  0.83   |   0.03   |   0.05   |    185   |
|          child_alone    |  0.00   |   0.00   |   0.00   |      0   |
|                water    |  0.88   |   0.12   |   0.21   |    350   |
|                 food    |  0.85   |   0.29   |   0.43   |    573   |
|              shelter    |  0.83   |   0.12   |   0.21   |    450   |
|             clothing    |  0.57   |   0.05   |   0.09   |     80   |
|                money    |  1.00   |   0.03   |   0.06   |    128   |
|       missing_people    |  0.00   |   0.00   |   0.00   |     64   |
|             refugees    |  0.00   |   0.00   |   0.00   |    162   |
|                death    |  0.68   |   0.06   |   0.11   |    213   |
|            other_aid    |  0.47   |   0.03   |   0.05   |    652   |
|infrastructure_related   |  0.80   |   0.01   |   0.02   |    362   |
|             transport   |  0.33   |   0.01   |   0.02   |    245   |
|             buildings   |  0.76   |   0.05   |   0.10   |    256   |
|           electricity   |  1.00   |   0.02   |   0.04   |    100   |
|                 tools   |  0.00   |   0.00   |   0.00   |     31   |
|            hospitals    |  0.00   |   0.00   |   0.00   |     59   |
|                shops    |  0.00   |   0.00   |   0.00   |     24   |
|          aid_centers    |  0.00   |   0.00   |   0.00   |     63   |
| other_infrastructure    |  0.33   |   0.00   |   0.01   |    248   |
|      weather_related    |  0.80   |   0.53   |   0.64   |   1437   |
|               floods    |  0.92   |   0.19   |   0.32   |    451   |
|                storm    |  0.70   |   0.21   |   0.32   |    473   |
|                 fire    |  0.00   |   0.00   |   0.00   |     44   |
|           earthquake    |  0.88   |   0.40   |   0.55   |    472   |
|                 cold    |  1.00   |   0.01   |   0.02   |    115   |
|        other_weather    |  0.67   |   0.01   |   0.02   |    261   |
|        direct_report    |  0.78   |   0.32   |   0.46   |   1010   |
|          avg / total    |  0.75   |   0.43   |   0.48   |  16358   |

### Future Opportunities
* Apply alternate classifiers: For this MVP, I only used the RandomForrestClassifier. I selected this model because it seems a reasonable approach to the issue. Other moddls which might be suited for this are: 
    * <a href='https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html'>SGDClassifier</a>
    * < href='https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svm#sklearn.svm.SVC>SVC</a>
* Add more hyper-parameters to the GridSearchCV function to see if a better preforming model might be identified. I did not do this for this MVP as the time to run the training script is already measured in minutes and each option added increased the time to run. I am already concerned other practicioners will exit out before the model selection process is finished. 
* Try to balance out the data somehow, perhaps with re-sampling or shaping the training data. In an ideal world, we would work with the provided to get a more balanced sample for learning purposes. At the very least, it would be nice to know which tags are most important and measure the model's performance against those tags. 

Questions or comments can be directed to the author. 
