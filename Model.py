# <<<<<<< HEAD
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn import metrics
# sns.set_theme()
# from IPython.display import HTML
# import plotly.express as px
# import warnings
# warnings.filterwarnings('ignore')
# calories = pd.read_csv("calories.csv")
# exercise = pd.read_csv("exercise.csv")
# exercise_df = exercise.merge(calories , on = "User_ID")
# exercise_df.drop_duplicates(subset = ['User_ID'], keep='last' , inplace = True)
# exercise_df.drop(columns = "User_ID" , inplace = True)
# exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.2 , random_state = 1)
# age_groups = ["Young" , "Middle-Aged" , "Old"]
# exercise_train_data["age_groups"] = pd.cut(exercise_train_data["Age"] , bins = [20 , 40 ,60 , 80] , right = False , labels = age_groups)
# for data in [exercise_train_data , exercise_test_data]:
#     data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
#     data["BMI"] = round(data["BMI"] , 2)

# bmi_category = ["Very severely underweight" , "Severely underweight" ,
#                 "Underweight" , "Normal" ,
#                 "Overweight" , "Obese Class I" ,
#                 "Obese Class II" , "Obese Class III"]

# exercise_train_data["Categorized_BMI"] = pd.cut(exercise_train_data["BMI"] , bins = [0 , 15 , 16 , 18.5 , 25 , 30 , 35 , 40 , 50]
#                                               , right = False , labels = bmi_category)

# exercise_train_data["Categorized_BMI"] = exercise_train_data["Categorized_BMI"].astype("object") 
# exercise_train_data = exercise_train_data[["Gender" , "Age" , "BMI" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
# exercise_test_data = exercise_test_data[["Gender" , "Age" , "BMI"  , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
# exercise_train_data = pd.get_dummies(exercise_train_data, drop_first = True)
# exercise_test_data = pd.get_dummies(exercise_test_data, drop_first = True)

# X_train = exercise_train_data.drop("Calories" , axis = 1)
# y_train = exercise_train_data["Calories"]

# X_test = exercise_test_data.drop("Calories" , axis = 1)
# y_test = exercise_test_data["Calories"]

# train_errors , val_errors = [] , []
# def plot_learning_curve(model):
#     for m in range(1 , 1000):
#         model.fit(X_train[:m] , y_train[:m])
#         y_train_predict = model.predict(X_train[:m])
#         y_val_predict = model.predict(X_test[:m])
#         train_errors.append(mean_squared_error(y_train[:m] , y_train_predict))
#         val_errors.append(mean_squared_error(y_test[:m] , y_val_predict))
# linreg = LinearRegression()
# linreg.fit(X_train , y_train)
# linreg_prediction = linreg.predict(X_test)

# random_reg = RandomForestRegressor(n_estimators = 1000 , max_features = 3 , max_depth = 6)
# random_reg.fit(X_train , y_train)
# random_reg_prediction = random_reg.predict(X_test)

# X_array = np.array([[21 , 25 , 50 , 100 , 40 , 0]]).reshape(1 , -1)
# y_pred = random_reg.predict(X_array)
# # print("Prediction : " , round(y_pred[0] , 2))



# import pickle

# # Save the Random Forest Regressor model to a .pkl file
# # model_filename = 'random_forest_model.pkl'
# with open('random_forest_model.pkl', 'wb') as file:
#     pickle.dump(random_reg, file)
# =======
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
sns.set_theme()
from IPython.display import HTML
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories , on = "User_ID")
exercise_df.drop_duplicates(subset = ['User_ID'], keep='last' , inplace = True)
exercise_df.drop(columns = "User_ID" , inplace = True)
exercise_train_data , exercise_test_data = train_test_split(exercise_df , test_size = 0.2 , random_state = 1)
age_groups = ["Young" , "Middle-Aged" , "Old"]
exercise_train_data["age_groups"] = pd.cut(exercise_train_data["Age"] , bins = [20 , 40 ,60 , 80] , right = False , labels = age_groups)
for data in [exercise_train_data , exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"] , 2)

bmi_category = ["Very severely underweight" , "Severely underweight" ,
                "Underweight" , "Normal" ,
                "Overweight" , "Obese Class I" ,
                "Obese Class II" , "Obese Class III"]

exercise_train_data["Categorized_BMI"] = pd.cut(exercise_train_data["BMI"] , bins = [0 , 15 , 16 , 18.5 , 25 , 30 , 35 , 40 , 50]
                                              , right = False , labels = bmi_category)

exercise_train_data["Categorized_BMI"] = exercise_train_data["Categorized_BMI"].astype("object") 
exercise_train_data = exercise_train_data[["Gender" , "Age" , "BMI" , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_test_data = exercise_test_data[["Gender" , "Age" , "BMI"  , "Duration" , "Heart_Rate" , "Body_Temp" , "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first = True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first = True)

X_train = exercise_train_data.drop("Calories" , axis = 1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories" , axis = 1)
y_test = exercise_test_data["Calories"]

train_errors , val_errors = [] , []
def plot_learning_curve(model):
    for m in range(1 , 1000):
        model.fit(X_train[:m] , y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_test[:m])
        train_errors.append(mean_squared_error(y_train[:m] , y_train_predict))
        val_errors.append(mean_squared_error(y_test[:m] , y_val_predict))
linreg = LinearRegression()
linreg.fit(X_train , y_train)
linreg_prediction = linreg.predict(X_test)

random_reg = RandomForestRegressor(n_estimators = 1000 , max_features = 3 , max_depth = 6)
random_reg.fit(X_train , y_train)
random_reg_prediction = random_reg.predict(X_test)

X_array = np.array([[21 , 25 , 50 , 100 , 40 , 0]]).reshape(1 , -1)
y_pred = random_reg.predict(X_array)
# print("Prediction : " , round(y_pred[0] , 2))



import pickle

# Save the Random Forest Regressor model to a .pkl file
# model_filename = 'random_forest_model.pkl'
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(random_reg, file)
# >>>>>>> f11e8731aa2246fe6f3b84af3b34c82d5aeaf924
