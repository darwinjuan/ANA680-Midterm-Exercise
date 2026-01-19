from flask import Flask, render_template, request
import pandas as pd
import statsmodels.api as sm  # Use the main API

app = Flask(__name__)

model = sm.load('car_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_values = [float(x) for x in request.form.values()]
        
        input_df = pd.DataFrame([input_values], 
                                columns=['Horsepower', 'Invoice', 'MPG_Highway'])
        
        prediction = model.predict(input_df)
        
        return render_template('index.html', 
                               prediction_text=f'Predicted City MPG: {prediction[0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)