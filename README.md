# realestate0

An end-to-end Data Science project that predicts property price per square foot in Mumbai, identifies underpriced and overpriced listings, and provides an interactive analytics dashboard using Streamlit.

This project combines web scraping, machine learning, geospatial analytics, and data visualization into a complete real-world property intelligence platform.

Project Highlights

Automated data collection using Scrapy
Data cleaning and historical price tracking
Feature engineering with real-world amenities
Outlier detection using Isolation Forest
Machine Learning model (XGBoost) for price prediction
Interactive Streamlit dashboard
Geospatial visualization using Folium
Identification of undervalued and overvalued properties

Tech Stack

Programming	- Python
Scraping - Scrapy
ML Models	- XGBoost, Scikit-learn
Data Processing	- Pandas, NumPy
Visualization	- Matplotlib, Folium
Geospatial - Geopy, OSMnx
Dashboard	- Streamlit
Deployment - GitHub

What This Project Solves

Instead of just showing property prices, this platform answers:

Is a property fairly priced?

Which listings are undervalued opportunities?

Which ones are overpriced?

How do amenities affect price?

What is the predicted price for a new property?

Machine Learning Approach
Target Variable
Price per Square Foot

Core Features Used
BHK
Bathrooms
Balconies
School density
Park density
Mall density
Hospital density
Historical suburb price trends
Model Used
XGBoost Regressor

Performance
RMSE: ~6800
R² Score: ~0.74

The model effectively captures location and amenity-driven pricing patterns.
