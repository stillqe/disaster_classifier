# Disaster Response Pipeline Project
### Project Overview
The project builds a web app that categorizes the type of emergency messages based on the needs communicated by the sender. 
This project is mainly composed of three components.
First is an ETL pipeline, which extracts data from CSV file, transforms raw data to the desired format, and loads data into a database. 
The second is a pipeline for natural language processing and machine learning, 
which normalizes and tokenizes texts, remove stop words, performs feature extraction such as Count Vector & TF-IDF, 
and finally classifies the messages using a random forest classifier with a grid search technique.
The last one is a part for deploying of web app, backed by Flask web framework. 
   
## File Descriptions
- app
    - run.py : start-up file to run the web app
    - wrangle_data.py: wangle data for visualizations in the web app
- data
    - process_data.py: ETL pipeline
    - disaster_categories.csv: contains disaster categories for each message
    - disaster_messages.csv: contains disaster messages
    - DisasterResponse.db: DB file saved from the ETL pipeline
- models
    - train_classifier.py: machine learning pipeline

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Licensing, Acknowledgements
This project is part of the requirements to complete Udacity's Data Science Nanodegree Program. 
The basic frame is provided by Udacity.