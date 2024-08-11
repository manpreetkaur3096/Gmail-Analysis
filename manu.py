import os
import pickle
import re
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import openai
import nltk
from nltk.corpus import stopwords
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if OPENAI_API_KEY is None:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Define the OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Download NLTK stopwords and cache it
@st.cache_resource
def download_nltk_stopwords():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = download_nltk_stopwords()

# Define the scope of access for the Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Authentication and API service
@st.cache_resource
def authenticate_user():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                r'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=8501)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
    return creds

@st.cache_resource
def get_gmail_service():
    creds = authenticate_user()
    service = build('gmail', 'v1', credentials=creds)
    return service

def get_latest_emails(service, max_results=10):
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=max_results).execute()
        messages = results.get('messages', [])
        emails = []

        for msg in messages:
            msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
            snippet = msg_data.get('snippet', '')
            emails.append(snippet)

        return emails
    except Exception as e:
        st.error(f"Failed to retrieve emails: {e}")
        return []

def clean_email_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def generate_word_cloud(emails):
    combined_text = ' '.join([clean_email_text(email) for email in emails])
    
    # Enhanced WordCloud with additional customizations
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        max_words=50, 
        background_color='black',    # Changed background color to black
        colormap='plasma',           # Set a vibrant colormap
        contour_color='steelblue',   # Add a contour around the words
        contour_width=2,             # Thickness of the contour
        font_path=None,              # You can specify a custom font path if desired
        prefer_horizontal=1.0,       # Prefer horizontal words
    ).generate(combined_text)
    
    return wordcloud


# Updated text summarizer function
def summarize_email(email_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Let me assist you"},
            {"role": "user", "content": f"Summary:\n\n{email_text[:2000]}"}
        ],
        temperature=0.5,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

# Updated sentiment analysis function
def analyze_sentiment(email_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Let me assist you"},
            {"role": "user", "content": f"Analyze the sentiment of the following email:\n\n{email_text[:2000]}\n\nSentiment:"}
        ],
        temperature=0,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

# Streamlit App
st.title("Gmail Analysis Tool")

# Sidebar options
option = st.sidebar.selectbox(
    'Choose an option',
    ('Generate Word Cloud', 'Email Summary & Sentiment Analysis')
)

# Authenticate and get the Gmail service
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if st.button('Authenticate with Gmail'):
    st.session_state.service = get_gmail_service()
    st.session_state.authenticated = True
    st.success("Successfully authenticated with Gmail!")

if st.session_state.authenticated:
    service = st.session_state.service
    if option == 'Generate Word Cloud':
        with st.spinner('Generating word cloud...'):
            spam_emails = get_latest_emails(service, max_results=30)
            if spam_emails:
                wordcloud = generate_word_cloud(spam_emails)
                st.image(wordcloud.to_array())
            else:
                st.write("No spam emails found.")

    elif option == 'Email Summary & Sentiment Analysis':
        with st.spinner('Retrieving latest emails...'):
            latest_emails = get_latest_emails(service, max_results=10)
            email_choice = st.selectbox('Select an email to analyze', latest_emails)
            if email_choice:
                with st.spinner('Analyzing selected email...'):
                    summary = summarize_email(email_choice)
                    sentiment = analyze_sentiment(email_choice)
                    st.write(f"**Summary:** {summary}")
                    st.write(f"**Sentiment:** {sentiment}")
