from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor

app = Flask(__name__)
CORS(app)
MODEL_PATH = 'savedModels/nn_model.pkl'
TRAINING_DATA_PATH = 'training_data.csv'
os.makedirs('savedModels', exist_ok=True)


def create_pipeline(X, y):
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json.get("input")
        if not input_data:
            return jsonify({'error': 'No input provided'}), 400

        df_input = pd.DataFrame([input_data])

        # Load or train model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
        else:
            if not os.path.exists(TRAINING_DATA_PATH):
                return jsonify({'error': 'No model and no training data available'}), 500

            df_train = pd.read_csv(TRAINING_DATA_PATH)
            df_train.dropna(inplace=True)

            if 'Price($)' not in df_train.columns:
                return jsonify({'error': "Target column 'Price($)' missing in training data"}), 400

            X = df_train.drop(columns=['Price($)'])
            y = df_train['Price($)']
            model = create_pipeline(X, y)
            joblib.dump(model, MODEL_PATH)

        # Predict
        prediction = model.predict(df_input)
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
