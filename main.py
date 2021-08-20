import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	data = pd.read_csv("Test_marks_data_collection.csv")

	data.rename(columns={'G1': 'Test1_score', 'G2': 'Test2_score', 'G3': 'Final_Test_score'}, inplace=True)
	X = data[['Test1_score', 'Test2_score']]
	y = data[["Final_Test_score"]]
	# data split
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
	model = LinearRegression()
	model.fit(X_train, y_train)
	prediction = model.predict(X_test)
	data2 = pd.DataFrame(np.c_[X_test, y_test, prediction],
						 columns=["Test1_score", "Test2_score", "Final_Test_score", "predcicted_Final_Test_score"])
	data2[data2["predcicted_Final_Test_score"] < 0] = 0



	if request.method == 'POST':
		test1 = request.form['test1']
		test2 = request.form['test2']
		data_sample = [[test1, test2]]
		my_prediction = model.predict(data_sample)

	predicted_value= str(my_prediction).lstrip('[[').rstrip(']]')




	return render_template('index.html',prediction = predicted_value)



if __name__ == '__main__':
	app.run(debug=True)

