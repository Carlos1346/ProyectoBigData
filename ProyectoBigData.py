import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

file_path = 'C:\\Users\\Chris\\OneDrive - Instituto Tecnológico de Aguascalientes\\ITA\\Materias\\Semestre 9\\Big data analytics\\Unidad 4\\dataset.csv'
data = pd.read_csv(file_path)

data.info()

#DATA PROCESSING
data.rename(columns = {'Nacionality':'Nationality', 'Age at enrollment':'Age'}, inplace = True)

#Null values
data.isnull().sum()/len(data)*100

print(data["Target"].unique())

data['Target'] = data['Target'].map({
    'Dropout':0,
    'Enrolled':1,
    'Graduate':2
})

print(data["Target"].unique())

data.corr()['Target']

plt.figure(figsize=(30, 30))
sns.heatmap(data.corr() , annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

new_data = data.copy()
new_data = new_data.drop(columns=['Unemployment rate','Inflation rate','GDP'], axis=1)
new_data = new_data.drop('Application order', axis=1)
new_data.info()

#EXPLORING DATA ANALYSIS
new_data['Target'].value_counts()

x = new_data['Target'].value_counts().index
y = new_data['Target'].value_counts().values

df = pd.DataFrame({
    'Target': x,
    'Count_T' : y
})

fig = px.pie(df,
             names ='Target', 
             values ='Count_T',
            title='How many dropouts, enrolled & graduates are there in Target column')

fig.update_traces(labels=['Graduate','Dropout','Enrolled'], hole=0.4,textinfo='value+label', pull=[0,0.2,0.1])
fig.show()

correlations = data.corr()['Target']
top_10_features = correlations.abs().nlargest(10).index
top_10_corr_values = correlations[top_10_features]

plt.figure(figsize=(10, 11))
plt.bar(top_10_features, top_10_corr_values)
plt.xlabel('Features')
plt.ylabel('Correlation with Target')
plt.title('Top 10 Features with Highest Correlation to Target')
plt.xticks(rotation=45)
plt.show()

px.histogram(new_data['Age'], x='Age',color_discrete_sequence=['lightblue'])

plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age', data=new_data)
plt.xlabel('Target')
plt.ylabel('Age')
plt.title('Relationship between Age and Target')
plt.show()

X = new_data.drop('Target', axis=1)
y = new_data['Target']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#BUILDING MODELS
dtree = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=2)
lr = LogisticRegression(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)

dtree.fit(X_train,y_train)
rfc.fit(X_train,y_train)
lr.fit(X_train,y_train)
knn.fit(X_train,y_train)

y_pred = dtree.predict(X_test)
print("Decision Tree Classifier accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")

y_pred = rfc.predict(X_test)
print("Random Forest Classifier accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")

y_pred = lr.predict(X_test)
print("Logistic Regression accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")

y_pred = knn.predict(X_test)
print("KNN accuracy :",round(accuracy_score(y_test,y_pred)*100,2),"%")

#SAVE THE BEST MODEL
import joblib

path = 'C:\\Apps\\ProyectoBigData'
model = 'dropout_model.pkl'
full_path = f"{path}/{model}"
joblib.dump(lr, full_path)
