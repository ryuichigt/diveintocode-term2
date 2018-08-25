class NN():
    def __init__(self,input_units,nn_units):
        np.random.seed(3)
        input_units = X.shape[1]
        self.w1 = np.random.randn(input_units,nn_units) / np.sqrt(2)
        self.w2 = np.random.randn(nn_units,2) /  np.sqrt(3)
        self.b1 = np.zeros((1,nn_units))
        self.b2 = np.zeros((1,2))
        self.param = { 'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}
        
    def step(self,x):
        if x > 0:
            return 1
        else:
            return 0
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def relu(self,x):
        return np.maximum(0,x)

    def tanh(self,x):
        e = np.exp(x)
        e_minus = np.exp(-x)
        result = (e-e_minus)/(e+e_minus)
        return result
    
    def softmax(self,a):
        c = np.max(a,axis = 0)
        e_a = np.exp(a)
        e_sum = np.sum(e_a,axis=1, keepdims=True)
        y = e_a/e_sum
        return y
    
    def forward_propagation(self):
        z1 = np.dot(self.x,self.w1) + self.b1
        a1 = tanh(z1)
        z2 = np.dot(a1,self.w2) + self.b2
        y = softmax(z2)
        return y

    def back_propagation(self,learning_rate=0.01):
        a1 = tanh(np.dot(self.x,self.w1) + self.b1)
        delta3 = (self.y_pred-np.identity(2)[self.y])#/len(y)
        delta2 = (1-a1**2) * np.dot(delta3,self.w2.T)
    
        self.w2 -= np.dot(a1.T,delta3)*learning_rate
        self.b2 -= np.sum(delta3,axis=0)*learning_rate
        self.w1 -= np.dot(self.x.T,delta2)*learning_rate
        self.b1 -= np.sum(delta2,axis=0)*learning_rate
        return self.w1,self.w2,self.b1,self.b2
    
    def fit(self,x,y,ite):
        self.x = x
        self.y = y
        self.y_pred = self.forward_propagation()
        
        for i in range(ite):
            
            self.w1,self.w2,self.b1,self.b2 = self.back_propagation()
            self.y_pred = forward_propagation(x,w1,w2,b1,b2)
            self.param = { 'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}
        return self.param
                      
    
    def predict(self,x):
        self.x = x
        pred = self.forward_propagation()
        return np.argmax(pred, axis=1)