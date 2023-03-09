
import numpy as np
import math


class LogisticRegression():
    def __init__(self):
        self.w = np.array([])
        self.loss_history = np.array([])
        self.score_history=np.array([])
    
    def fit(self,X,y, alpha = 0.001, max_epochs =100):
        n,p = X.shape
        self.w = np.random.rand(p+1)
        done = False
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        curr_epoch = 0
        
        while (not done):
            gradient = self.gradient(X_,y)
            self.w -=alpha*gradient
            new_loss = self.loss(X_,y)
           
            self.loss_history = np.append(self.loss_history,new_loss)
            self.score_history = np.append(self.score_history, self.score(X_,y))
            
            if (np.isclose(gradient.all(),0)) or (curr_epoch>=max_epochs):
                done =True
            
            curr_epoch+=1
    
    def score(self,X,y):
        return (y== self.predict(X)).mean()
    
    def gradient(self,X,y):
        sigmoid = self.sigmoid(X@self.w)
        return np.mean(((sigmoid - y)[:,np.newaxis]*X),axis=0)
    
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, X,y):
        y_hat = X@self.w
        return (-y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))).mean()

        
    def predict(self, X):
        return 1*(X@self.w)>0
    
    def fit_stochastic(self, X,y,batch_size = 10, alpha=0.001,max_epochs=100):
        n,p = X.shape
        self.w = np.random.rand(p+1)
        done = False
        X_ = np.append(X, np.ones((X.shape[0],1)),1)
        
        for j in np.arange(max_epochs):
            
            order = np.arange(n)
            np.random.shuffle(order)
            prev_loss = np.inf
            
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X_[batch,:]
                y_batch = y[batch]
                grad = self.gradient(x_batch, y_batch) 
                self.w -=alpha*grad

                new_loss = self.loss(X_,y)


                if (np.isclose(new_loss,0) or np.isnan(new_loss)):
                    break
                if new_loss<prev_loss:
                    prev_loss = new_loss
            self.loss_history = np.append(self.loss_history,new_loss)
            self.score_history = np.append(self.score_history, self.score(X_,y))

            
                    
        
    
    
#     def __init__(self):
#         self.w=[] #track the weights
#         self.loss_history=np.array([]) #track the loss
#         self.score_history=np.array([]) #track the score
    
#     def fit(self,X,y, alpha = 0.001, max_epochs = 100):
#         n,p = X.shape
#         self.w = np.random.rand(p+1)
#         done = False
#         X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
#         curr_epoch=0
#         while (done ==False) and (curr_epoch<max_epochs):
#             self.w-= alpha*self.gradient(X_,y)
#             new_loss = self.loss(X_,y) 
#             self.loss_history = np.append(self.loss_history,new_loss)
#             self.score_history = np.append(self.score_history,self.score(X_,y))
            
#             if np.allclose(self.gradient(X_,y),0):
#                 done = True
#             curr_epoch+=1

        
#     def gradient(self,X,y):
#         n = X.shape[1]
#         predictions = self.predict(X)
#         calcs= X*(self.sigmoid(predictions)-y)[:,np.newaxis]
#         return np.mean(calcs,axis=0)

         
    
#     def loss(self,X, y):
#         y_hat = self.predict(X)
#         return (-y*np.log(self.sigmoid(y_hat)) - (1-y)*np.log(1-self.sigmoid(y_hat))).mean()
     
#     def sigmoid(self,z):
#         return 1/(1+np.exp(-z))
    
#     def predict(self,X):
        
#         return ((X@self.w))
    
#     def score(self,X,y):
#         return (y== (1*(X@self.w)>0)).mean() 

#     ####################   STOCHASTIC DESCENT  ###########################
    
#     def fit_stochastic(self, X,y, alpha = .001, m_epochs = 100, batch_size = 15):
        
#         X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
#         n = X_.shape[0]

#         self.w = np.random.rand(X_.shape[1])
   
#         prev_loss= np.inf
        
#         for j in np.arange(m_epochs):
#             order = np.arange(n)
#             np.random.shuffle(order)

#             for batch in np.array_split(order, n // batch_size + 1):
#                 x_batch = X_[batch,:]
#                 y_batch = y[batch]
#                 self.w-= alpha*self.gradient(x_batch,y_batch)

#                 new_loss = self.loss(x_batch,y_batch) 
          
#             if np.isclose(new_loss,0):
#                 break
#             else:
#                 prev_loss = new_loss
                    
#             self.loss_history = np.append(self.loss_history,self.loss(X_,y))
#             self.score_history = np.append(self.score_history,self.score(X_,y))