import pandas as pd
import plotly.express as px
import pandas as pd
import csv
import plotly.graph_objects as go

df = pd.read_csv("./116/data_classification.csv")


hours_slept = df["Hours_Slept"].tolist()
hours_studied = df["Hours_studied"].tolist()



#Plotting the data on the scatter plot
fig = px.scatter(x=hours_slept, y=hours_studied)
fig.show()
#----------------------------------------------------------
#plotting the scatter plot with all the variables to see how the different variables affect the results
import plotly.graph_objects as go

hours_slept = df["Hours_Slept"].tolist()
hours_studied = df["Hours_studied"].tolist()

results = df["results"].tolist()
colors=[]
for data in results:
  if data == 1:
    colors.append("green")
  else:
    colors.append("red")



fig = go.Figure(data=go.Scatter(
    x=hours_studied,
    y=hours_slept,
    mode='markers',
    marker=dict(color=colors)
))
fig.show()

#---------------------------------------------------

#Splitting the data into two parts . Using a part of this data to train the prediction model and other part to test the prediction model.

#hours studied and slept of the person
hours = df[["Hours_studied", "Hours_Slept"]]

#results
results = df["results"]

#splitting the data into 75% and 25%. 75% data for training and then we will test our model on the remaining 25% percent of the data to test and determine the accuracy of our model.


#-----------------------------------------------------

from sklearn.model_selection import train_test_split 

hours_train, hours_test, results_train, results_test = train_test_split(hours, results, test_size = 0.25, random_state = 0)
print(hours_train)

#We don't need to scale the vlaues as they are already same.

#Training the logistic regression model

from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(hours_train, results_train)

LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)


#Testing the prediction model

results_pred = classifier.predict(hours_test)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(results_test, results_pred)) 

#The accuracy we got is 0.92 which is very impressive.

#Now lets test the model .


from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
hours_train = sc_x.fit_transform(hours_train)  

user_hours_studied = int(input("Enter hours studied -> "))
user_hours_slept = int(input("Enter hours slept -> "))

user_test = sc_x.transform([[user_hours_studied, user_hours_slept]])

user_result_pred = classifier.predict(user_test)

if user_result_pred[0] == 1:
  print("This user may pass!")
else:
  print("This user may not pass!")

  #output
  #Enter hours studied -> 7
#Enter hours slept -> 12
#This user may not pass!