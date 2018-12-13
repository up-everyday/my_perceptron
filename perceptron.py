"""
Perceptron Algorithm from Scratch.

The first part of this code is training and testing a Perceptron algorithm 
from scratch.

The final part compares the results from a scikit-learn Perceptron with the 
algorithm implemented from scratch in the first part.
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("dataset.csv")

"""
Taking a look at the data using a scatterplot.
It's clearly linearly separable.
"""
obs = 1000
plt.figure(0) #0 -> title
plt.scatter(df.values[:,1], df.values[:,2], c = df['3'], alpha=0.8) #c ->color, sequence, or sequence of color, optional
plt.show()



"""
Splitting the data into train/test sets. 70% train/30% test.
This is good practice to avoid overfitting.
"""
# .values converts the dataframe to numpy array. Makes it easier to shuffle and split
df = df.values

np.random.seed(5)
np.random.shuffle(df) # shuffles only rows:
train = df[0:int(0.7*len(df))] #rows
test = df[int(0.7*len(df)):int(len(df))]

x_train = train[:, 0:3]
y_train = train[:, 3]

x_test = test[:, 0:3]
y_test = test[:, 3]



"""
Training the perceptron.
The output will be the weights (w) and the sum-of-squared error loss function (J).
"""
def perceptron_train(x, y, z, eta, t):
    '''
    Input Parameters:
        x: data set of input features
        y: actual outputs
        z: activation function threshold
        eta: learning rate
        t: number of iterations
    '''

    # initializing the weights
    w = np.zeros(len(x[0]))
    n = 0

    # initializing additional parameters to compute sum-of-squared errors
    yhat_vec = np.ones(len(y))     # vector for predictions
    errors = np.ones(len(y))       # vector for errors (actual - predictions)
    J = []                         # vector for the SSE cost function

    while n < t:
        for i in range(0, len(x)):

            # dot product
            f = np.dot(x[i], w)

            # activation function
            if f >= z:
                yhat = 1.
            else:
                yhat = 0.
            yhat_vec[i] = yhat

            # updating the weights
            for j in range(0, len(w)):
                w[j] = w[j] + eta*(y[i]-yhat)*x[i][j]

        n += 1
        # computing the sum-of-squared errors
        #for i in xrange(0,len(y)): #        #for i in xrange(0,len(y)):
        for i in range(0, len(y)):
           errors[i] = (y[i]-yhat_vec[i])**2
        J.append(0.5*np.sum(errors))

    return w, J

z = 0.0
eta = 0.1
t = 50

perceptron_train(x_train, y_train, z, eta, t)

w = perceptron_train(x_train, y_train, z, eta, t)[0]
J = perceptron_train(x_train, y_train, z, eta, t)[1]

print("w is ",w)
print("J is ", J)



"""
Here's a plot of the convergence of the perceptron.
You can change the threshold (z), learning rate (eta), and number of iterations
(t) and see how that affects the convergence.
"""
J = perceptron_train(x_train, y_train, z, eta, t)[1]
epoch = np.linspace(1,len(J),len(J))

plt.figure(1)
plt.plot(epoch, J)
plt.xlabel('Epoch')
plt.ylabel('Sum-of-Squared Error')
plt.title('Perceptron Convergence')
plt.show()


"""
Testing the model using the accuracy score from sklearn.metrics
"""
from sklearn.metrics import accuracy_score

w = perceptron_train(x_train, y_train, z, eta, t)[0]

def perceptron_test(x, w, z, eta, t):
    y_pred = []
    for i in range(0, len(x-1)):
        f = np.dot(x[i], w)

        # activation function
        if f > z:
            yhat = 1
        else:
            yhat = 0
        y_pred.append(yhat)
    return y_pred

y_pred = perceptron_test(x_test, w, z, eta, t)

print ("The accuracy score is:")
print (accuracy_score(y_test, y_pred))



"""
Plotting the decision boundary.
The formula for the decision boundary is:
    0 = w0x0 + w1x1 + w2x2

We can rearrange this and solve for x2.
x2 = (-w0x0-w1x1)/w2
"""

min = np.min(x_test[:,1])
max = np.max(x_test[:,1])
x1 = np.linspace(min,max,100)

def x2(x1, w):
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    x2 = []
    for i in range(0, len(x1-1)):
        x2_temp = (-w0-w1*x1[i])/w2
        x2.append(x2_temp)
    return x2

x_2 = np.asarray(x2(x1,w))

df = pd.read_csv("dataset.csv")

obs = 1000
plt.figure(2)
plt.scatter(df.values[:,1], df.values[:,2], c = df['3'], alpha=0.8)
plt.plot(x1, x_2)
plt.show()


"""
Making a prediction with the sklearn Perceptron.
We want to compare the weights with this model to the weights from our model
above.

We need to set the following to make sure that we're using the same data
as what we used in the model above:
    random_state=None
    eta0=0.1 (learning rate)
    shuffle=False
    fit_intercept=False (we're already fitting the intercept because we
                         included a dummy feature of ones in the dataset)
"""
from sklearn.linear_model import Perceptron

# training the sklearn Perceptron
clf = Perceptron(random_state=None, eta0=0.1, shuffle=False, fit_intercept=False, max_iter=1000, tol=0.19)
clf.fit(x_train, y_train)
y_predict = clf.predict(x_test)

print ("sklearn weights:")
print (clf.coef_[0])

print ("my perceptron weights:")
print (w)

