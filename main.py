from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
app = Flask(__name__)
model = pickle.load(open("model.pkl", 'rb'))

#defining column name
cols = [ 'Month',
       'Year','Sectors', 'Total Customers']

  
@app.route('/')

def index():
    return render_template('index.html')

@app.route("/predict",methods=['POST'])
def prediction():
    if request.method == 'POST':
        Sectors = float(request.form['sectors'])
        Month= float(request.form['month'])
        Year= float(request.form['year'])
        Total_Customers= float(request.form['tcustomer'])
        sector_names = {
            0: 'LV Agricultural Customers',
            1: 'LV Bulk Customers',
            2: 'LV Commercial Customers',
            3: 'LV Domestic Customers',
            4: 'LV Electric Cars',
            5: 'LV Cottage & Small Industry',
            6: 'LV Highlanders',
            7: 'LV Industrial Customers',
            8: 'LV Institutional Customers',
            9: 'LV Power House Customers',
            10: 'LV Religious Institution Customers',
            11: 'LV Rural Customers',
            12: 'LV Rural Domestic Customers',
            13: 'LV Rural-Community Lhakhangs',
            14: 'LV Rural-Micro Trade',
            15: 'LV Rural-Cooperatives and Agriculture',
            16: 'LV Street Light Customers',
            17: 'LV Temporary Customers',
            18: 'LV Urban Domestic Customers',
            19: 'MV Industrial Customers',
            
        }
        sector_name = sector_names.get(Sectors, 'Unknown Sector')

        x_sample = [[ Month,
       Year,Sectors, Total_Customers]]
        X = pd.DataFrame(x_sample,columns=cols)
        result = model.predict(X)
        
        return render_template("result.html",value=result, sector_name=sector_name)
    
if __name__=="__main__":
    app.run(debug=True)
