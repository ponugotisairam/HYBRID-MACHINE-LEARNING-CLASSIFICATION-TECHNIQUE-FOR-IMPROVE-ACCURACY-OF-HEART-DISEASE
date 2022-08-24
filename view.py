from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def index(request):
    return render(request,'home.html')

def proceed(request):
    global data
    if request.method=='POST':
        file = request.FILES['filename']
        data=pd.read_csv(file)
        data['bmi'].fillna(data['bmi'].median(), inplace=True)
        data.drop('id', axis=1, inplace=True)
        data['gender'].replace(['Male', 'Female', 'Other'], [0, 1, 2], inplace=True)
        data['ever_married'].replace(['Yes', 'No'], [0, 1], inplace=True)
        data['work_type'].replace(['Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked'], [0, 1, 2, 3, 4],inplace=True)
        data['Residence_type'].replace(['Urban', 'Rural'], [0, 1], inplace=True)
        data['smoking_status'].replace(['never smoked', 'Unknown', 'formerly smoked', 'smokes'], [0, 1, 2, 3],inplace=True)
        for i in ['avg_glucose_level']:
            q3, q1 = np.percentile(data.loc[:, i], [75, 25])
            IQR = q3 - q1
            max = q3 + (1.5 * IQR)
            min = q1 - (1.5 * IQR)
            data.loc[data[i] < min, i] = np.nan
            data.loc[data[i] > max, i] = np.nan
            data['avg_glucose_level'].fillna(data['avg_glucose_level'].median(), inplace=True)
        for i in ['bmi']:
            q3, q1 = np.percentile(data.loc[:, i], [75, 25])
            IQR = q3 - q1
            max = q3 + (1.5 * IQR)
            min = q1 - (1.5 * IQR)
            data.loc[data[i] < min, i] = np.nan
            data.loc[data[i] > max, i] = np.nan
            data['bmi'].fillna(data['bmi'].median(), inplace=True)

            col = data.columns
            rows=data.values.tolist()
            return render(request,'showdata.html',{'cols':col,'rows':rows})
    return render(request,'index.html')

def modelselection(request):
    global X_train, X_test, y_train, y_test,model_h,auc,rfc_acc,accuracy,acc
    if request.method=='POST':
        X = data.drop('stroke', axis=1)
        y = data['stroke']
        os = SMOTE()
        X, y = os.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model=request.POST['models']
        if model == '1':
            model_n = DecisionTreeClassifier()
            model_n.fit(X_train, y_train)
            pred = model_n.predict(X_test)
            auc = accuracy_score(pred, y_test)
            return render(request,'modelselection.html',{'acc':auc})
        elif model == '2':
            rfc = RandomForestClassifier(n_estimators=90)
            rfc.fit(X_train, y_train)
            pred1 = rfc.predict(X_test)
            rfc_acc = accuracy_score(pred1, y_test)
            print(rfc_acc)
            return render(request, 'modelselection.html', {'acc': rfc_acc})
        elif model=='3':
            model = XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_pred, y_test)
            return render(request, 'modelselection.html', {'acc': accuracy})
        elif model == '4':
            level0 = list()
            level0.append(('DT', DecisionTreeClassifier()))
            level0.append(('XGBoost', XGBClassifier()))
            level0.append(('RF', RandomForestClassifier(n_estimators=200)))
            model_h=StackingClassifier(estimators=level0)
            model_h.fit(X_train, y_train)
            y = model_h.predict(X_test)
            acc = accuracy_score(y_test, y)
            return render(request, 'modelselection.html', {'acc':acc})
        else:
            print("hellooooooooooooooo")
    return render(request,'modelselection.html')


def graph(request):
    x = ['Decision Tree', 'Random Forest','Hybrid Model', 'XGB']
    y = [auc,rfc_acc,acc,accuracy]
    graph = sns.barplot(x, y)
    plt.title('Model Accuracies')
    graph.set(ylabel="Accuracy")
    graph.set(xlabel="Models")
    plt.show()
    return render(request,'predict_val.html')


def predict(request):
    if request.method=='POST':
        gender = request.POST['gender']
        age = request.POST['age']
        hypertension=request.POST['hypertension']
        heart_disease=request.POST['heart_disease']
        ever_married=request.POST['ever_married']
        work_type=request.POST['work_type']
        Residence_type=request.POST['Residence_type']
        avg_glucose_level=request.POST['avg_glucose_level']
        bmi=request.POST['bmi']
        smoking_status=request.POST['smoking_status']
        x=[[float(gender),float(age),float(hypertension),float(heart_disease),float(ever_married),float(work_type),float(Residence_type),float(avg_glucose_level),float(bmi),float(smoking_status)]]
        print(x)
        y=pd.DataFrame(x,columns = X_train.columns)
        y_pred = model_h.predict(y)
        print(y_pred)
        if y_pred==[1]:
            messages='Their is a chance to get stroke'
            return render(request, 'predict_val.html',{'a':messages})
        else:
            messages='Their is no chance to get stroke'
            return render(request, 'predict_val.html',{'a':messages})
    return render(request,'predict_val.html')
