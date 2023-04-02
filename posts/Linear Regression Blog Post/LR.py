
import numpy as np

from sklearn.metrics import r2_score

class LinearRegression():
    
    def __init__(self):
        self.w = np.array([])
        self.score_history = np.array([])

    
    def pad(self,X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
        
    
    def fit_analytic(self,X,y):
        X_ = self.pad(X)
        self.w = np.linalg.inv(np.transpose(X_)@X_)@np.transpose(X_)@y
    
    
        
    def fit_grad(self,X,y,alpha = .005, max_epochs = 1000):
        X_ = self.pad(X)
        n,p = X.shape
        done = False
        self.w = np.random.rand(p+1,1)
        curr_epoch = 0
        
        P = np.transpose(X_)@X_
        q = (np.transpose(X_)@y)
        q = q.reshape(q.shape[0],1)
        
        while (not done):
            gradient = (P@self.w - q)
            self.w -=alpha*gradient
            self.score_history = np.append(self.score_history, self.score(X_,y))
            if (np.isclose(gradient.all(),0)) or (curr_epoch>=max_epochs):
                done =True
            curr_epoch+=1
    
    def predict(self,X):
        
        w = self.w.reshape(self.w.shape[0])
     
        return X@w
    
    def score(self,X,y):
        y_hat = self.predict(X)
        y_bar = y.mean()
        ss_tot = ((y-y_bar)**2).sum()
        ss_res = ((y-y_hat)**2).sum()
        return (1 - (ss_res/ss_tot))
        
        
        
        

    

        

        
            
        
        
    
        
        
        