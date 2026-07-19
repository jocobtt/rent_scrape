# Models Backend API

## Overview

This repository contains the backend API for a model that analyzes the rental market in Japan, specifically focusing on apartments in Tokyo. The API serves predictions based on various input features related to apartment listings, allowing users to estimate rental prices effectively.

## Purpose

The primary goal of this API is to provide a user-friendly interface for accessing machine learning models that predict rental prices based on input features such as:

- **rei_price**: The initial cost of renting the apartment.
- **shikikin**: The deposit required.
- **maintenence_price**: Monthly maintenance fees.
- **sqr_m**: The size of the apartment in square meters.
- **year_built**: The age of the building.
- **floor**: The floor number of the apartment.
- **eki_walk**: The walking distance to the nearest station.

## Decisions Made

1. **Model Selection**: We opted for a linear regression model due to its simplicity and interpretability, which is suitable for our initial analysis. As the project evolves, we may explore more complex models to improve accuracy.

2. **Data Handling**: The data is scraped from the Suumo website, a popular apartment listing platform in Japan. We ensure that the data is cleaned and preprocessed to handle missing values and convert categorical variables into numerical formats.

3. **Deployment**: The API is built using FastAPI, which allows for high performance and easy integration with machine learning models. We use BentoML for model serving, enabling us to deploy our models efficiently.

4. **Version Control**: We utilize MLflow for tracking experiments, logging metrics, and managing model versions. This helps in maintaining a clear history of model performance and facilitates easy rollback if needed.

## How It Works

1. **Data Scraping**: The data is collected using a web scraping script that extracts relevant information from the Suumo website. The scraped data is then cleaned and stored in a structured format.

2. **Model Training**: The cleaned data is used to train the linear regression model. The training process involves splitting the data into training and testing sets, fitting the model, and evaluating its performance using metrics like Mean Squared Error (MSE).

3. **API Endpoints**:
   - **/predict**: This endpoint accepts a JSON payload with the required input features and returns the predicted rental price.
   - **/challenger_predict**: This endpoint accepts a JSON payload with the required input features and returns the predicted rental price.
   - **/health**: A simple health check endpoint to verify that the API is running correctly.

4. **Deployment**: The API is deployed on Google Cloud Run, allowing for scalable and serverless execution. The deployment process is automated using GitHub Actions, ensuring that the latest changes are reflected in production seamlessly.

## Getting Started

To run the API locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

4. Access the API at `http://localhost:8000`.

## Future Improvements

- **Model Enhancement**: Explore more advanced models and techniques to improve prediction accuracy.
- **Testing**: Implement comprehensive testing strategies to ensure the reliability of the API.
- **Documentation**: Expand the documentation to include detailed API specifications and usage examples.
- **Retraining api endpoint**: Endpoint for retraining
- Better metric gathering 
- Connect to scraper to pull down more recent data 
- include feature engineering. 

