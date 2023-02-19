import random 
import numpy as np

class Perceptron:
    
    
    def __init__(self):
        """
        initialize Perceptron class with a weight and history array
        """
        self.w = []
        self.history =[]
        
    def perceptron_classify(self,x):
        """
        classifies a single data point
        
        inputs: 
            w: weight array
            x: data matrix
            
        returns:
            true (1) or false (-1) for the predicted label of one data point
        """

        return 2*(self.w@x > 0)-1 # turns T or F to int values 1 and -1

        
    def score(self,X,y):
        """
        purpose finds the accuracy of the model at its current stage
        
        inputs: 
            X_: data with an added column vector of 1s at the end
            y_: labels in the form of -1 (false) and 1 (true)
            
        returns:
            the accuracy of the perceptron as a number between 0 and 1
            with 1 corresponding to perfect accuracy
        """
        
        y_ = 2*y-1 
        num_matches = (y_==(2*self.predict(X)-1)).mean() 
        return num_matches
    
    
    def predict(self,X):
        """
        inputs:
            X_: data with an added column vector of 1s at the end
            
        returns:
            a vector y^ in {0,1} of predicted labels
        """
        
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        return (1*((X_@self.w)>0))
        
    
    def fit(self,X,y,maxsteps = 1000):
        
        """
        fits a perceptron to the data iteratively updating the weights 
        until it achieves 100% accuracy or the maximum number of steps
        inputs: 
            X: array of features
            y: array of binary labels  {0,1}
            maxsteps: the maximum number of steps/updates allowed
        return:
            None
        """
        
        n,p = X.shape # get number of data points and features
        
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1) #add last column of 1s
        y_ = 2*y-1 #changes y from 0 and 1 to -1 and 1
        self.w = np.random.rand(p+1) #generate a random array of weights

        accuracy = 0 #
        steps =0
       
        while accuracy!= 1 and steps<maxsteps:
    
            random_int = np.random.randint(0,n-1) #generate a random integer index
            prediction = self.perceptron_classify(X_[random_int]) # predcit the label 

            
            self.w = self.w + 1*(prediction*y_[random_int]<0)*y_[random_int]*X_[random_int] #update weights
            
            accuracy = self.score(X,y) #calculate loss
            self.history.append(accuracy)
            steps+=1
            
        
        