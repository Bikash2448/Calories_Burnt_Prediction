<<<<<<< HEAD
import numpy as np
from flask import Flask,request ,render_template , jsonify
import pickle
app = Flask(__name__)

model = pickle.load(open('random_forest_model.pkl','rb'))

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def calculate():
    if request.method=='POST':
        features = [int(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = round(prediction[0] , 2)
        return render_template('results.html', results = output)




if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
=======
import numpy as np
from flask import Flask,request ,render_template , jsonify
import pickle
app = Flask(__name__)

model = pickle.load(open('random_forest_model.pkl','rb'))

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def calculate():
    if request.method=='POST':
        features = [int(x) for x in request.form.values()]
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = round(prediction[0] , 2)
        return render_template('index.html', results = output)




if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
>>>>>>> f11e8731aa2246fe6f3b84af3b34c82d5aeaf924
