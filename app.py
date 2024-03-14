from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import ast

# Import libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from itertools import product

items = []

#Use optimized parameters
sgd_model = SGDRegressor(
    loss='squared_error',
    penalty = 'l1',
    max_iter=1000, 
    tol=1e-3, 
    random_state=42,
    alpha = 0.1,
    epsilon = 0.01,
    learning_rate = 'invscaling',
    eta0 = 0.06,
    power_t = 0.25,
    validation_fraction = 0.1,
    early_stopping = False,
    n_iter_no_change = 5,
)

# Initialize the MLPRegressor with the best parameters
mlp_model = MLPRegressor(
    hidden_layer_sizes=(35,),  # one hidden layer with 35 neurons
    activation='relu',
    solver='adam',
    alpha=0.3,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=5,
    verbose=False
)

# Initialize the SVRegressor with the best parameters
svr_model = SVR(
    kernel = 'rbf',
    C = 10,
    gamma = 0.01,
    epsilon = 0.1,
    tol = 1e-3,
    max_iter = -1
)

# Load the forest fires dataset
df_forest_fires = pd.read_csv('forest+fires/forestfires.csv')
items.append(str(df_forest_fires.head(5).to_html(classes='table table-striped', index=False)))

# Generate the heatmap
df_forest_fires_cont_vars = df_forest_fires.drop(columns = ['X', 'Y', 'month', 'day'])
cont_vars_corr_matrix = df_forest_fires_cont_vars.corr()
heatmap_fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.heatmap(cont_vars_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.close(heatmap_fig1)

# Save the first heatmap as bytes
heatmap_bytes1 = BytesIO()
heatmap_fig1.savefig(heatmap_bytes1, format='png')
heatmap_bytes1.seek(0)
items.append(base64.b64encode(heatmap_bytes1.read()).decode('utf-8'))
heatmap_bytes1.close()

# Do the necessary preprocessing
df_forest_fires['fire_occurred'] = df_forest_fires['area'].apply(lambda x: 1 if x > 0 else 0)
df_forest_fires['log_area'] = np.log1p(df_forest_fires['area'])

# Remove outliers
Q1 = df_forest_fires['log_area'].quantile(0.25)
Q3 = df_forest_fires['log_area'].quantile(0.75)
IQR = Q3 - Q1

is_outlier = (df_forest_fires['log_area'] < (Q1 - 1.5 * IQR)) | (df_forest_fires['log_area'] > (Q3 + 1.5 * IQR))
df_updated = df_forest_fires[~is_outlier]
df_updated.reset_index(drop=True, inplace=True)

# Define output and target variables
X = df_updated.drop(['area', 'log_area', 'fire_occurred'], axis=1)
y = df_updated[['area', 'log_area']]

# Preform additional preprocessing
selected_features = ['temp', 'RH', 'wind', 'rain']

# One-hot encoding for categorical vars ('month', 'day')
categorical_features = ['month', 'day']
one_hot_encoder = OneHotEncoder()
preprocessor = ColumnTransformer(
    transformers=[('cat', one_hot_encoder, categorical_features)],
    remainder='passthrough'
)
X_transform = preprocessor.fit_transform(X)
scaler = StandardScaler()
scaler.fit_transform(X_transform)
X_transform = preprocessor.transform(X)
X_scale = scaler.transform(X_transform)

# Fit the models
sgd_model.fit(X_scale, y['log_area'])
mlp_model.fit(X[selected_features], y['log_area'])
svr_model.fit(X[selected_features], y['log_area'])

items.append(str(X.columns.tolist()))
items.append(str(X[selected_features].columns.tolist()))
items.append(str(X[selected_features].columns.tolist()))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

class Form(FlaskForm):
    name = StringField('Feature Values')
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', items=items)

@app.route('/SGD', methods=['GET', 'POST'])
def sgd():
    form = Form()
    if form.validate_on_submit():
        string = form.name.data
        print(string)
        observation = ast.literal_eval(string)
        print(observation)
        df = pd.DataFrame([observation], columns=X.columns.tolist())
        print(df.head())
        df_transform = preprocessor.transform(df)
        df_scale = scaler.transform(df_transform)
        prediction = sgd_model.predict(df_scale)
        print(prediction)
        answer = math.exp(prediction[0]) - 1
        print(answer)
        return render_template('sgd.html', items=items, form=form, answer=answer)
    return render_template('sgd.html', items=items, form=form, answer=None)

@app.route('/MLP', methods=['GET', 'POST'])
def mlp():
    form = Form()
    if form.validate_on_submit():
        string = form.name.data
        print(string)
        observation = ast.literal_eval(string)
        print(observation)
        df = pd.DataFrame([observation], columns=selected_features)
        print(df.head())
        prediction = mlp_model.predict(df)
        print(prediction)
        answer = math.exp(prediction[0]) - 1
        print(answer)
        return render_template('mlp.html', items=items, form=form, answer=answer)
    return render_template('mlp.html', items=items, form=form, answer=None)

@app.route('/SVR', methods=['GET', 'POST'])
def svr():
    form = Form()
    if form.validate_on_submit():
        string = form.name.data
        print(string)
        observation = ast.literal_eval(string)
        print(observation)
        df = pd.DataFrame([observation], columns=selected_features)
        print(df.head())
        prediction = svr_model.predict(df)
        print(prediction)
        answer = math.exp(prediction[0]) - 1
        print(answer)
        return render_template('svr.html', items=items, form=form, answer=answer)
    return render_template('svr.html', items=items, form=form, answer=None)

if __name__ == '__main__':
    app.run(debug=True)