# Japanese Apartment Rent Analysis 🗼

Welcome to the Japanese Apartment Rent Analysis project, a data-driven solution designed to offer valuable insights into the rental market in Japan. This project scrapes data from [Suumo](https://suumo.jp/), a leading real estate platform in Japan, and utilizes machine learning to predict apartment rental prices. The project aims to help users identify the best value options available in the Japanese rental market.

### Project Overview
This repository contains a comprehensive analysis tool that combines data science, machine learning, and web development technologies to deliver accurate rent predictions and informative visualizations for apartment seekers, property managers, and researchers.

## Key Features
- Data Collection: Automated data scraping from Suumo to collect relevant apartment listings and rental information.
- Machine Learning Models: Implementation of two core models:
    - A baseline regression model for initial rent predictions.
    - A challenger model that uses advanced regression techniques, providing more refined predictions and facilitating continuous improvement.
- Deployment: Models are serialized with joblib and served via a FastAPI backend for real-time predictions.
- Interactive Visualizations: Data visualizations built with D3.js to help users intuitively explore market trends.

## Technical Stack
- Frontend: Built with Vue/Vite and deployed on Vercel for a seamless and interactive user experience.
- Backend: FastAPI powers the backend, serving predictions and handling data processing.
- Machine Learning: Models are trained and validated using regression techniques to achieve high prediction accuracy.
- Deployment: Both frontend and backend components are hosted on GitHub and Vercel.

## Project Structure
This project is divided into two repositories:

- Backend: The FastAPI backend and machine learning model code can be found here in this repository.
- Frontend: The D3.js data visualizations and Vue frontend code can be found [here](https://github.com/jocobtt/rent_d3).

## Getting Started
- Clone the Repository: Clone the backend and frontend repositories.
- Install Dependencies: Use the provided requirements.txt for the backend and package.json for the frontend to install dependencies.
- Run the Application: Start the FastAPI server and Vue frontend, following the instructions in each repository's README.

## Future Enhancements
- Model Expansion: Integrate additional machine learning models, including deep learning approaches.
- Improved Visualizations: Expand D3.js visualizations to include more granular insights and filtering options.
- Enhanced Scraping: Regularly update the scraping pipeline to ensure data accuracy and freshness. 


