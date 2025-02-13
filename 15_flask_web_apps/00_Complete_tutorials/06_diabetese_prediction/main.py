from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__, template_folder='template')
svm_model=pickle.load(open('svm_model.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

def std_scalar(df):
    std_X = StandardScaler()
    x =  pd.DataFrame(std_X.fit_transform(df))
    return x

def pipeline(features):
    steps = [('scaler', StandardScaler()), ('SVM', svm_model)]
    pipe = Pipeline(steps)
    return pipe.fit_transform(features)


@app.route('/send', methods=['POST'])
def getdata():

    features = [float() for x in request.form.values()]
    final_features = [np.array(features)]

    #Feature tranform and prediction using pipeline
    # We can now use predictions from this feature_tranformed variable
    #feature_tranformed= pipeline(final_features)


    feature_transform=std_scalar(final_features)
    # Using standard scalar method
    prediction = svm_model.predict(feature_transform)
    if prediction==0:
        result="You Are Non-Diabetic"
    else:
        result="You Are Diabetic"

    Pregnancies=request.form['Pregnancies']
    Glucose = request.form['Glucose']
    BloodPressure = request.form['BloodPressure']
    SkinThickness = request.form['SkinThickness']
    Insulin = request.form['Insulin']
    BMI = request.form['BMI']
    DiabetesPedigreeFunction = request.form['DiabetesPedigreeFunction']
    Age = request.form['Age']
    return render_template('show.html', preg=Pregnancies, bp=BloodPressure,
                           gluc=Glucose, st=SkinThickness, ins=Insulin, bmi=BMI,
                           dbf=DiabetesPedigreeFunction, age=Age, res=result)


if __name__=="__main__":
    app.run(debug=True)