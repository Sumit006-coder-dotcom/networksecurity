# Network Security Threat Detection System

## Overview
The *Network Security Threat Detection System* is a machine learning-based project designed to analyze network traffic and detect potential malicious activities. It helps network administrators monitor and secure their networks by classifying network events as *Normal* or *Malicious*.

This project leverages *data preprocessing, **feature imputation, and **machine learning models* to predict network threats in a structured and reliable way.

## Features
- *Network Threat Detection:* Detects malicious network activity using machine learning.
- *Data Preprocessing:* Handles missing or incomplete data with KNNImputer.
- *Custom Exception Handling:* Provides clear error messages for debugging.
- *Modular Design:* Uses structured Python classes and modules for maintainability.
- *Extensible:* Can be integrated into real-time network monitoring systems.

2. Install dependencies:
pip install -r requirements.txt

Usage

1. Prepare network data in the expected format (CSV or JSON) with the correct features.

2. Run the application:

uvicorn app:app --reload

3. The system will process the input, handle missing features, and predict whether the network activity is Normal or Malicious.

3. Error Handling

1.The system raises NetworkSecurityException if there are issues like:
2.Missing or extra features
3.Model loading errors
4.Invalid input data

Example:

X has 110 features, but KNNImputer is expecting 111 features as input.

4. Dependencies

Python >= 3.8

pandas

numpy

scikit-learn

pymongo

python-dotenv

certifi

Install all dependencies using:

pip install -r requirements.txt

# Future Enhancements

Real-time network monitoring with live packet capture.

Integration with dashboards for alerts and analytics.

Support for more machine learning algorithms to improve accuracy.


# Author

Sumit Kumar Karn
Email: sumitkarn2005@gmail.com
GitHub: https://github.com/Sumit006-coder-dotcom
LinkedIn: https://www.linkedin.com/in/sumit-karn-86606524a/