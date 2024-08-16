import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
import os

# Path to the user file
user_file_path = 'users/data/users.txt'

# Create the directory and file if they don't exist
os.makedirs(os.path.dirname(user_file_path), exist_ok=True)
if not os.path.exists(user_file_path):
    with open(user_file_path, 'w') as f:
        pass  # Create the empty file

# Load users from the file
def load_users():
    with open(user_file_path, 'r') as f:
        return f.read().splitlines()

# Save a new user to the file
def save_user(username):
    with open(user_file_path, 'a') as f:
        f.write(username + '\n')

# Initial configuration
if 'username' not in st.session_state:
    st.session_state['username'] = ""

# Prompt for username at the start
def prompt_for_username():
    st.title("Welcome to the Reading Recommendation Tool")
    st.text("Please choose an option. No personal data is collected.")
    
    option = st.radio("Select an option", ("Sign in with existing user", "Create a new user"))

    if option == "Sign in with existing user":
        users = load_users()
        username = st.selectbox("Select your username", users)
        if st.button("Sign in"):
            if username:
                st.session_state['username'] = username
                st.session_state['read_articles'] = []
            else:
                st.error("Please select a valid username.")
    elif option == "Create a new user":
        new_username = st.text_input("Enter a username")
        if st.button("Create user"):
            if new_username:
                users = load_users()
                if new_username not in users:
                    save_user(new_username)
                    st.session_state['username'] = new_username
                    st.session_state['read_articles'] = []
                    st.success(f"User {new_username} created successfully.")
                else:
                    st.error("The username already exists. Please choose another one.")
            else:
                st.error("Please enter a valid username.")

# Show the prompt at the start if no username is set
if not st.session_state['username']:
    prompt_for_username()

# Verify if a username has been assigned
if st.session_state['username']:
    # Load the data
    df = pd.read_csv('./data/guardian_articles_full_content.csv')
    df = df.dropna(subset=['bodyText'])
    df['webPublicationDate'] = pd.to_datetime(df['webPublicationDate'])

    # Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['bodyText'])
    cosine_similarities = cosine_similarity(tfidf_matrix)

    # Configuration of read articles
    if 'read_articles' not in st.session_state:
        st.session_state['read_articles'] = []

    # Function to recommend articles
    def recommend_articles(article_id, num_recommendations=5):
        article_index = df.index[df['id'] == article_id][0]
        sim_scores = list(enumerate(cosine_similarities[article_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        recommended_articles = [df.iloc[i] for i, _ in sim_scores]
        return recommended_articles

    # Function to show instructions
    def show_instructions():
        st.title("Instructions and Guide")
        st.write("""
        Welcome to the Reading Recommendation Tool.
        
        1. **Search**: Here you can search for articles by section and title. You can also mark articles as read.
        2. **Recommendation System**: Get recommendations based on the articles you've read.
        3. **Your Articles**: A list of all the articles you have read so far.
        4. **Reset Preferences**: Clear all read articles history and restart the tool.
        """)

    # Search section
    def search_module():
        st.title('Guardian News Recommendation System - Search')

        section = st.selectbox("Choose your preferred news section:", df['sectionName'].unique())
        search_query = st.text_input("Search for a news article by title:")

        filtered_df = df[(df['sectionName'] == section) & 
                         (df['webTitle'].str.contains(search_query, case=False))]

        st.subheader('Filtered Articles')
        for i in range(min(5, len(filtered_df))):
            article = filtered_df.iloc[i]
            read_status = "✅ " if article['id'] in st.session_state['read_articles'] else ""
            st.write(f"{read_status}**{article['webTitle']}**")
            st.write(f"*Published on:* {article['webPublicationDate']}")
            st.write(article['bodyText'][:200] + '...')
            st.write(f"[Read more]({article['webUrl']})")

            if article['id'] not in st.session_state['read_articles']:
                if st.button(f"Mark as Read - {article['webTitle']}", key=f"read-{article['id']}"):
                    st.session_state['read_articles'].append(article['id'])
            st.write('---')

    # Recommendation System section
    def recommendation_system_module():
        st.title('Guardian News Recommendation System - Recommendation System')

        if len(st.session_state['read_articles']) == 0:
            st.write("Please read an article in the 'Search' tab to generate recommendations.")
        else:
            st.subheader("Articles You Might Like:")
            for article_id in st.session_state['read_articles']:
                recommended_articles = recommend_articles(article_id)
                for article in recommended_articles:
                    st.write(f"**{article['webTitle']}**")
                    st.write(f"*Published on:* {article['webPublicationDate']}")
                    st.write(article['bodyText'][:200] + '...')
                    st.write(f"[Read more]({article['webUrl']})")
                    st.write('---')

    # Your Articles section
    def your_articles_module():
        st.title('Your Articles')

        if len(st.session_state['read_articles']) == 0:
            st.write("You haven't read any articles yet.")
        else:
            for article_id in st.session_state['read_articles']:
                article = df[df['id'] == article_id].iloc[0]
                st.write(f"**{article['webTitle']}**")
                st.write(f"*Published on:* {article['webPublicationDate']}")
                st.write(article['bodyText'][:200] + '...')
                st.write(f"[Read more]({article['webUrl']})")
                st.write('---')

    # Function to reset preferences
    def reset_preferences():
        st.session_state['read_articles'] = []
        st.session_state['username'] = ""
        st.experimental_rerun()

    # Navigation menu
    with st.sidebar:
        st.title("Navigation")
        selection = option_menu(
            menu_title="",
            options=["Instructions and Guide", "Search", "Recommendation System", "Your Articles"],
            icons=["info-circle", "search", "bar-chart-fill", "book"],
            menu_icon="cast",
            default_index=0,
        )

        if st.button("Reset Preferences"):
            reset_preferences()

    # Navigation logic
    if selection == "Instructions and Guide":
        show_instructions()
    elif selection == "Search":
        search_module()
    elif selection == "Recommendation System":
        recommendation_system_module()
    elif selection == "Your Articles":
        your_articles_module()
