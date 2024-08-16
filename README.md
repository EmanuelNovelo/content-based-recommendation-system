# Content-Based News Recommendation System

A content-based recommendation system uses natural language processing and machine learning to suggest articles to users based on the content they have previously read.

## Project Overview

This project implements a content-based recommendation system that fetches news articles from The Guardian API, processes them using machine learning techniques, and serves personalized article recommendations through a Streamlit web app. The app allows users to log in, mark articles as liked or disliked, and provides personalized recommendations.

## Project Workflow

1. **Environment Setup**:
   - Configuration of a Python development environment using Visual Studio Code.
   - Installation of necessary packages, including `pandas`, `scikit-learn`, `Streamlit`, and `nltk` for natural language processing.

2. **Data Collection**:
   - Fetch news articles using The Guardian API.
   - Store articles' metadata and content for further processing.

3. **Data Processing**:
   - Clean and preprocess the article data, focusing on key fields like title, body, and publication date.
   - Use `TF-IDF` (Term Frequency-Inverse Document Frequency) to vectorize the text content.
   - Calculate cosine similarity between articles to determine relatedness.
   - Use `nltk` to include synonyms in the search functionality, enhancing the search experience.

4. **Recommendation System Development**:
   - Implementation of a content-based filtering approach to recommend articles similar to those previously read by the user.
   - Allow users to provide feedback (like/dislike) to improve future recommendations.

5. **Deployment and User Interaction**:
   - Serve the model through a Streamlit app.
   - Provide a user interface for searching articles, viewing recommendations, and tracking reading history.
   - Include user login functionality, allowing multiple users to maintain separate preferences.

## Technologies Used

- **Python**: Main programming language for the project.
- **Streamlit**: For building the interactive web application.
- **The Guardian API**: For fetching news articles.
- **pandas & NumPy**: For data manipulation and analysis.
- **scikit-learn**: For machine learning tasks, including TF-IDF vectorization and cosine similarity.
- **nltk**: For natural language processing, including handling synonyms.
- **Matplotlib**: For visualizing user feedback statistics.

## Project Goals

- Demonstrate the ability to build and deploy a content-based recommendation system.
- Provide a user-friendly web application for interacting with the recommendation system.
- Implement user feedback mechanisms to refine and personalize recommendations.

## How to Run

1. Clone the repository to your local machine.
2. Set up the environment with the necessary packages as described.
3. Fetch the latest articles from The Guardian using the provided scripts.
4. Run the Streamlit app to start the recommendation system.
5. Explore, search, and get recommendations based on your reading history.

## Future Work

- Enhance the model by incorporating user feedback (likes/dislikes) into the recommendation algorithm.
- Explore additional machine learning algorithms for improving recommendations.
- Implement a more sophisticated user management system.
