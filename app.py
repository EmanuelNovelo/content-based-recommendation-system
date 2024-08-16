import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu
import nltk
from nltk.corpus import wordnet
import os
import matplotlib.pyplot as plt

# Cache the TF-IDF and similarity calculations to speed up the app
@st.cache_data
def get_tfidf_and_similarities(df):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['bodyText'])
    cosine_similarities = cosine_similarity(tfidf_matrix)
    return tfidf_vectorizer, cosine_similarities

# Function to expand search query with synonyms
def expand_query_with_synonyms(query):
    synonyms = set([query])
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    return " | ".join(synonyms)

# Path to the user file
user_file_path = 'users/data/users.txt'
os.makedirs(os.path.dirname(user_file_path), exist_ok=True)
if not os.path.exists(user_file_path):
    with open(user_file_path, 'w') as f:
        pass

def load_users():
    with open(user_file_path, 'r') as f:
        return f.read().splitlines()

def save_user(username):
    with open(user_file_path, 'a') as f:
        f.write(username + '\n')

# Initial configuration
if 'username' not in st.session_state:
    st.session_state['username'] = ""

if 'read_articles' not in st.session_state:
    st.session_state['read_articles'] = []

if 'liked_articles' not in st.session_state:
    st.session_state['liked_articles'] = []

if 'disliked_articles' not in st.session_state:
    st.session_state['disliked_articles'] = []


def prompt_for_username():
    st.title("Welcome to the Content-Based Recommendation Tool for The Guardian News")
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

if not st.session_state['username']:
    prompt_for_username()

if st.session_state['username']:
    df = pd.read_csv('./data/guardian_articles_full_content.csv')
    df = df.dropna(subset=['bodyText'])
    df['webPublicationDate'] = pd.to_datetime(df['webPublicationDate'])

    tfidf_vectorizer, cosine_similarities = get_tfidf_and_similarities(df)

    if 'read_articles' not in st.session_state:
        st.session_state['read_articles'] = []

    def recommend_articles(article_id, num_recommendations=5):
        article_index = df.index[df['id'] == article_id][0]
        sim_scores = list(enumerate(cosine_similarities[article_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:num_recommendations + 1]
        recommended_articles = [df.iloc[i] for i, _ in sim_scores]
        return recommended_articles

    def show_user_and_instructions():
        st.title("User & Instructions")
        st.write(f"**Current User:** {st.session_state['username']}")
        st.write("""
        Welcome to the Content-Based Recommendation Tool.
        
        1. **Search**: Here you can search for articles by section and title. You can also mark articles as read.
        2. **Recommendation System**: Get recommendations based on the articles you've read.
        3. **Your Articles**: A list of all the articles you have read so far.
        4. **Reset Preferences**: Clear all read articles history and restart the tool.
        
        *Remember to visit the 'Recommended Artciles' section to provide your feedback by liking the articles!*
        """)

        if st.button("Log out or change user"):
            st.session_state['username'] = ""
            st.session_state['read_articles'] = []
            st.stop()

    def search_module():
        st.title('Guardian News Contend-Based Recommendation System - Search')

        section = st.selectbox("Choose your preferred news section:", ["All Sections"] + df['sectionName'].unique().tolist())
        search_query = st.text_input("Search for a news article by title:")

        if search_query:
            expanded_query = expand_query_with_synonyms(search_query)
            st.write(f"Expanded query: {expanded_query}")  # This will display the expanded query
        else:
            expanded_query = ""

        filtered_df = df if section == "All Sections" else df[df['sectionName'] == section]
        filtered_df = filtered_df[filtered_df['webTitle'].str.contains(expanded_query, case=False)]

        st.subheader('Filtered Articles')
        for i in range(min(5, len(filtered_df))):
            article = filtered_df.iloc[i]
            read_status = "✅ " if article['id'] in st.session_state['read_articles'] else ""
            st.write(f"{read_status}**{article['webTitle']}**")
            st.write(f"*Published on:* {article['webPublicationDate']}")
            st.write(article['bodyText'][:200] + '...')
            st.write(f"[Read more]({article['webUrl']})")

            if article['id'] not in st.session_state['read_articles']:
                if st.button(f"Mark as Read", key=f"read-{article['id']}"):
                    st.session_state['read_articles'].append(article['id'])
            st.write('---')

    def recommendation_system_module():
        st.title('Guardian News Recommendations')

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
                    if st.button(f"I like this!", key=f"like-{article['id']}"):
                        st.session_state['liked_articles'].append(article['id'])
                    if st.button(f"I don't like this!", key=f"dislike-{article['id']}"):
                        st.session_state['disliked_articles'].append(article['id'])
                    st.write('---')

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
                st.write(f"[Read more]({article['webUrl']})")  # Added "Read more" link
                st.write('---')


    def stats_module():
        st.title('User Feedback Statistics')

        total_likes = len(st.session_state.get('liked_articles', []))
        total_dislikes = len(st.session_state.get('disliked_articles', []))

        st.subheader("Your Feedback Summary:")
        st.write(f"**Total 'I like this!' Articles:** {total_likes}")
        st.write(f"**Total 'I don't like this!' Articles:** {total_dislikes}")
        
        # Display a pie chart only if there's feedback
        if total_likes > 0 or total_dislikes > 0:
            feedback_labels = ['Likes', 'Dislikes']
            feedback_counts = [total_likes, total_dislikes]

            st.write("### Feedback Distribution")
            st.pyplot(create_feedback_pie_chart(feedback_labels, feedback_counts))
        else:
            st.write("No feedback data available to display.")

    def create_feedback_pie_chart(labels, counts):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        
        # Ensure counts are valid (i.e., no NaN values)
        counts = [count if not pd.isna(count) else 0 for count in counts]
        
        # Create the pie chart only if there are non-zero counts
        if sum(counts) > 0:
            ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')

        return fig


    def reset_preferences():
        st.session_state['read_articles'] = []
        st.stop()

    with st.sidebar:
        st.title("Navigation")
        selection = option_menu(
            menu_title="",
            options=["User & Instructions", "Search", "Recommended Articles", "Your Articles", "Stats"],
            icons=["person-circle", "search", "bar-chart-fill", "book", "graph-up"],
            menu_icon="cast",
            default_index=0,
        )

        if st.button("Reset Preferences"):
            reset_preferences()

    if selection == "User & Instructions":
        show_user_and_instructions()
    elif selection == "Search":
        search_module()
    elif selection == "Recommended Articles":
        recommendation_system_module()
    elif selection == "Your Articles":
        your_articles_module()
    elif selection == "Stats":
        stats_module()
