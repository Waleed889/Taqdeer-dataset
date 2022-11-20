from flask import Flask, request, render_template
import pandas as pd
import pickle
import os 
from os.path import join, dirname, realpath
import json 
import joblib

app = Flask(__name__)
print('#'*50,dirname(realpath(__file__)))
UniqueKeys = json.load(open(join(dirname(realpath(__file__)),'UniqueKyes.json')))
Parts = json.load(open(join(dirname(realpath(__file__)),'parts.json')))

# Loading our model 
model = joblib.load(join(dirname(realpath(__file__)), 'XGBR_Model_19_23-21'))


@app.route('/')
def home():

    return render_template('home.html', UniqueKeys = UniqueKeys, Parts=Parts)

@app.route('/predict', methods=['POST'])
def predict():

    input_dict = {'Area': 2,
 'CarBrand': 92,
 'CarModel': 201,
 'ManufactureYear': 2016,
 'PaymentType': 0,
 'CarMade': 8,
 'CarType': 1,
 'PartsNumber': 1,
 'PartOfDay': 0,
 'part_Bridge': 0,
 'part_Bumper': 0,
 'part_Coilover': 0,
 'part_Control Arms': 0,
 'part_Dash insulator': 0,
 'part_Decoration': 0,
 'part_Door': 0,
 'part_Fender': 0,
 'part_Fiber': 0,
 'part_Grill': 0,
 'part_Handle': 0,
 'part_Headlight': 0,
 'part_Hinges': 0,
 'part_Hood': 0,
 'part_Injection': 0,
 'part_Mirror': 0,
 'part_Mudguard': 0,
 'part_Muffler': 0,
 'part_Other': 0,
 'part_Power Window': 0,
 'part_Radiator': 0,
 'part_Rims': 0,
 'part_Rotor': 0,
 'part_Sensor': 0,
 'part_Shock absorber': 0,
 'part_Splash shield': 0,
 'part_Stabilizer link': 0,
 'part_Taillight': 0,
 'part_Tie rod': 0,
 'part_Tire': 0,
 'part_Windshild ': 0,
 'pos_front': 0,
 'pos_front left': 0,
 'pos_front right': 0,
 'pos_left': 0,
 'pos_rear': 0,
 'pos_rear left': 0,
 'pos_rear right': 0,
 'pos_right': 0,
 'pos_undefined': 0,
 'state_New': 0,
 'state_Used': 0}




    # Geting features and values from the usern 
    data = request.form.to_dict()
    
    # Separ
    print(data)
    for k, v in data.items():
        if k == 'Parts':
            if v in input_dict:
                input_dict[v]=1
            else:
                for k2,v2 in input_dict.items():
                    try:
                        if k2.split('_')[1].split(' ')[0] == v:
                            input_dict[k2]=1
                    except:
                        continue

        elif k == 'Position':
            if v in input_dict:
                input_dict[v]=1
            else:
                for k2,v2 in input_dict.items():
                    if k2.split('_')[1].split(' ')[0] == v:
                        input_dict[k2]=1
        elif k == 'State':
            input_dict[v]=1
        else:
            input_dict[k]=v

            

    user_df = pd.DataFrame(input_dict, index=[0])

    print(user_df)
    # Make predictions on dataframe
    prediction = model.predict(user_df)
    # get the output
    output = prediction[0]

    return render_template('predictions.html', prediction= output, data=data)

if __name__ == "__main__":
    app.run(debug=True)