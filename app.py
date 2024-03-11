from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import OneClassSVM, SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import product

items = []

# Load the forest fires dataset
df_forest_fires = pd.read_csv('forest+fires/forestfires.csv')
items.append(str(df_forest_fires.head(5).to_html(classes='table table-striped', index=False)))

# Generate the heatmap
df_forest_fires_cont_vars = df_forest_fires.drop(columns = ['X', 'Y', 'month', 'day'])
cont_vars_corr_matrix = df_forest_fires_cont_vars.corr()
heatmap_fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.heatmap(cont_vars_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.close(heatmap_fig1)

# Save the first heatmap as bytes
heatmap_bytes1 = BytesIO()
heatmap_fig1.savefig(heatmap_bytes1, format='png')
heatmap_bytes1.seek(0)
items.append(base64.b64encode(heatmap_bytes1.read()).decode('utf-8'))
heatmap_bytes1.close()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

class MyForm(FlaskForm):
    name = StringField('Name')
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def home():

    form = MyForm()

    if form.validate_on_submit():
        name = form.name.data
        greeting = f"Hello, {name}!"
        return render_template('index.html', items=items, form=form, greeting=greeting)

    return render_template('index.html', items=items, form=form, greeting=None)

if __name__ == '__main__':
    app.run(debug=True)