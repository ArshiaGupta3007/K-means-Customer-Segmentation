from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

app = Flask(__name__)

# Generate sample data for clustering
np.random.seed(42)
data = {
    'CustomerID': np.arange(1, 101),
    'Age': np.random.randint(18, 65, size=100),
    'Average_Spend': np.random.uniform(5, 50, size=100),
    'Visits_per_Week': np.random.uniform(1, 7, size=100),
    'Promotion_Interest': np.random.randint(1, 11, size=100)
}
df = pd.DataFrame(data)

# Prepare the data for clustering
features = df[['Age', 'Average_Spend', 'Visits_per_Week', 'Promotion_Interest']]
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

# Rename clusters to 'Daily', 'Promotion', 'Weekend'
cluster_names = {0: 'Daily', 1: 'Promotion', 2: 'Weekend'}
df['Customer Group'] = df['Cluster'].map(cluster_names)

# Define a function for clustering
def clustering(age, avg_spend, visit_per_week, promotion_interest):
    new_customer = np.array([[age, avg_spend, visit_per_week, promotion_interest]])
    predicted_cluster = kmeans.predict(new_customer)
    return cluster_names[predicted_cluster[0]]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data.get('age')
    avg_spend = data.get('avg_spend')
    visit_per_week = data.get('visit_per_week')
    promotion_interest = data.get('promotion_interest')
    
    cluster = clustering(age, avg_spend, visit_per_week, promotion_interest)
    return jsonify({'cluster': cluster})

if __name__ == '__main__':
    app.run(debug=True)
