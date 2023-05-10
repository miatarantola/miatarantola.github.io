from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class DatasetClass:

    def train_test_data(self, dataset):
        """
       This function is made to split data into training and testing data sets. It is split into 2/3 training data,
       1/3 testing data. 
       It is specific to our cleaned data set pollution_income_race.
        """
        dataset = dataset[dataset["City"] != "Not in a city"]

        dataset["AQI Binary"] = 1 * (dataset["AQI Total"] <= 100)

        #our labels are the AQI Binary for now
        y = dataset.loc[:,"AQI Binary"]

        #also going to drop states and cities for now, (and city) because we don't get a numerical from that
        dataset = dataset.drop(columns=["State_x", 
                                  "County_x",
                                  "City",
                                   "State_y", 
                                  "County_y", 
                                  "AQI Total", 
                                  "AQI Binary"])

        #finally, drop those features that directly contribute to AQI (i.e. NO2 Mean, NO2 1st Max Value)
        dataset = dataset.drop(columns=["NO2 Mean",
                                 "NO2 1st Max Value",
                                 "NO2 1st Max Hour",
                                 "NO2 AQI",
                                 "O3 Mean",
                                 "O3 1st Max Value",
                                 "O3 1st Max Hour",
                                 "O3 AQI",
                                 "SO2 Mean",
                                 "SO2 1st Max Value",
                                 "SO2 1st Max Hour",
                                 "SO2 AQI",
                                 "CO Mean",
                                 "CO 1st Max Value",
                                 "CO 1st Max Hour",
                                 "CO AQI"])

        #train and test data
        X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.33, random_state=42)

        return X_train, X_test, y_train, y_test    
    
