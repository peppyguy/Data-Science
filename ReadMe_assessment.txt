There are 4 files:

- Fake_GET.py, generates the REST webservice which returns a fake resolution date 

- Training.py, runs the training algorithm and writes this model into a file called AVRO-RForest-prediction.sav

- Prediction_GET.py, runs the REST webservice and returns the predicted resolution date for the requested issue. Before running this file we need to run first the Training.py file.

- Release_plan.py: runs the REST web service and returns the issues with predicted resolution date that agrees with the required date in the GET request. Before running this file we need to run  Training.py. The possible format for the date in the GET request is of the type year-month or year-month-day. It is possible to generalise this to include additional information about hours, etc, though precision of a day is already very significant.

 NOTE: path for the data files json and csv is 'data/avro*.*' . I am assuming that the script is in the same folder as the data folder.