import pandas as pd
import plotly.express as px
import pandas as pd
import csv
import plotly.graph_objects as go

df = pd.read_csv("./116/logistics_data.csv")
#Here, we have loaded the data. Let's see how the data looks like in a scatter plot -
salary = df["EstimatedSalary"].tolist()
purchased = df["Purchased"].tolist()

print(len(salary))

fig = px.scatter(x=salary, y=purchased)
fig.show()

#-----------------------------------------------------------------
#Let's plot the data on the scatter plot to see how different variables effect the purchase.
import plotly.graph_objects as go

salaries = df["EstimatedSalary"].tolist()
ages = df["Age"].tolist()

purchased = df["Purchased"].tolist()
colors=[]
for data in purchased:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")



fig = go.Figure(data=go.Scatter(
    x=salaries,
    y=ages,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()
#observation:We can see that the plot is split into two parts , the red dots represent the people who haven't bought the phone and the green dots represent the people who have bought the phone.


#Now using the machine learning libraries we'll build a model to predict if the person will buy a phone or not. To do that we'll divide the data into two parts . We'll use the first part to train this model to predict if person will buy the phone or not. And use the second part to test our model.

#Dividing the data and using a part of it to train the model on it and then using the other part of the data to test the model is a general practice used by data scientists.

#-------------------------------------------------------------------------
#Let's do that! We will consider the age of the person and the salary to determine if they will purchase the product or not.

#Taking together Age and Salary of the person
factors = df[["EstimatedSalary", "Age"]]

#Purchases made
purchases = df["Purchased"]

#Out of the data that we have, we will use 75% data for training and then we will test our model on the remaining 25% percent of the data to test and determine the accuracy of our model.


#Let's start by splitting the data.
from sklearn.model_selection import train_test_split 

salary_train, salary_test, purchase_train, purchase_test = train_test_split(factors, purchases, test_size = 0.25, random_state = 0)

#Now, we have divided our data successfully in 75% 25% ratios.


#If we look at the data, we are considering the age and the salary together. While the age of the person is in years, the salary might be in INR or Dollars. We have to make sure that they are using the same unit before we proceed further.


#For this, we are going to use sklearn's StandardScaler. This will compare values with each other, and determine a score for the values. Both the age and the salary of the person will be given a score in comparison to others. Let's see how our data changes after doing this.

print(salary_train[0:10])

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 

salary_train = sc_x.fit_transform(salary_train)  
salary_test = sc_x.transform(salary_test) 
  
print (salary_train[0:10])

#Here, we can see that both the age and the salary of the person are given points. Now, we are sure that each and every feature will contribute equally to the decision making of our machine learning model, where we want to predict if the user will buy the product or not.

#--------------------------------------------------------------
#Now, let's train a logistic regression model on the training data that we seperated out earlier. For this, we will use ```sklearn```'s pre-built class ```LogisticRegression```.

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(salary_train, purchase_train)


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
#Here, the ```random_state = 0``` means that we are providing the model with the Training data. If the value would have been 1, it would mean that we are providing with the testing data.


#Now that our model for Logistic Regression is trained, it's time for us to test this model. Is it predicting values well? Let's find out.

purchase_pred = classifier.predict(salary_test)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(purchase_test, purchase_pred)) 

#An accuracy of 1 would have been perfectly accurate, 0.89 is an excellent accuracy.


#Let's try to predict if a user will buy the product or not, using this model.

user_age = int(input("Enter age of the customer -> "))
user_salary = int(input("Enter the salary of the customer -> "))

user_test = sc_x.transform([[user_salary, user_age]])

user_purchase_pred = classifier.predict(user_test)

if user_purchase_pred[0] == 1:
  print("This customer may purchase the product!")
else:
  print("This customer may not purchase the product!")


  #output
  #Enter age of the customer -> 23
#Enter the salary of the customer -> 120000
#his customer may not purchase the product!

