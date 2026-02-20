

<h1 align="center">🌍 Air Quality Index (AQI) Prediction System</h1>

<p align="center">
A Machine Learning–based Air Quality Management System that predicts AQI using historical pollutant data and presents meaningful insights through an interactive web application.
</p>

<hr>

<h2>📌 Project Overview</h2>
<ul>
  <li>Predict AQI using historical air pollution data</li>
  <li>Classify AQI into standard categories</li>
  <li>Provide health advisory messages</li>
  <li>Show year-based and city-wise analysis</li>
  <li>Interactive Streamlit web application</li>
</ul>

<hr>

<h2>🚀 System Workflow</h2>
<ol>
  <li>User selects a <b>City</b></li>
  <li>User selects a <b>Year</b></li>
  <li>System retrieves historical pollutant data</li>
  <li>Machine Learning model predicts AQI</li>
  <li>Displays:
    <ul>
      <li>AQI Value</li>
      <li>AQI Category</li>
      <li>Health Advisory Message</li>
      <li>Visual Insights & Trends</li>
    </ul>
  </li>
</ol>

<hr>

<h2>📊 AQI Category Standard</h2>

<table border="1" cellpadding="8">
<tr>
<th>AQI Range</th>
<th>Category</th>
</tr>
<tr><td>0–50</td><td>Good</td></tr>
<tr><td>51–100</td><td>Satisfactory</td></tr>
<tr><td>101–200</td><td>Moderate</td></tr>
<tr><td>201–300</td><td>Poor</td></tr>
<tr><td>301–400</td><td>Very Poor</td></tr>
<tr><td>401–500</td><td>Severe</td></tr>
</table>

<hr>

<h2>✨ Key Features</h2>
<ul>
  <li>City-based AQI Prediction</li>
  <li>Year-based AQI Analysis</li>
  <li>AQI Category Classification</li>
  <li>Health Impact Messages</li>
  <li>City-wise Pollution Trends</li>
  <li>Interactive Web Interface (Streamlit)</li>
</ul>

<hr>

<h2>🛠 Technology Stack</h2>
<ul>
  <li><b>Python</b></li>
  <li>Pandas & NumPy</li>
  <li>Scikit-learn (Machine Learning)</li>
  <li>Streamlit (Web Application)</li>
  <li>GitHub (Version Control)</li>
</ul>

<hr>

<h2>🔄 Overall System Flow</h2>

<pre>
Historical Dataset
        ↓
Data Cleaning
        ↓
ML Model Training
        ↓
AQI Prediction
        ↓
AQI Categorization + Health Advisory
        ↓
Web Application Display
        ↓
Insights & Evaluation
</pre>

<hr>

<h2>📂 Project Structure</h2>

<pre>
AQI-Prediction-System/
│
├── data/
│   └── cleaned_air_quality.csv
│
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── model_training.ipynb
│   ├── analysis.ipynb
│   └── evaluation.ipynb
│
├── assets/
│   └── graphs & images
│
├── app.py
├── model.pkl
├── aqi_utils.py
└── README.md
</pre>

<hr>

<h2>👥 Team Members</h2>

<table border="1" cellpadding="8">
<tr>
<th>Name</th>
<th>Branch</th>
<th>Role</th>
</tr>
<tr><td>Ayush</td><td>CSE Core</td><td>Team Lead & ML Model</td></tr>
<tr><td>Avinash</td><td>EIE</td><td>Data Cleaning</td></tr>
<tr><td>Rohith</td><td>ECE</td><td>Analysis & Insights</td></tr>
<tr><td>Hiten</td><td>CSE DS</td><td>Web Application</td></tr>
<tr><td>Rasika</td><td>CSE DS</td><td>AQI Logic & Integration</td></tr>
</table>

<hr>

<h2>📌 Project Statement</h2>

<p>
"We are developing a machine-learning–based air quality management system that predicts AQI using historical pollutant data and presents year-based analysis and health insights through an interactive web application."
</p>

<hr>

<p align="center">
⭐ If you like this project, consider giving it a star on GitHub!
</p>
