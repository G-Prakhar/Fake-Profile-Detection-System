# streamlit_fake_profile_detector.py
# A self-contained Streamlit app demonstrating a simple ML-based fake profile detector.
# Run: pip install -r requirements.txt
# Then: streamlit run streamlit_fake_profile_detector.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import joblib
import io

st.set_page_config(page_title="Fake Profile Detector", layout="wide")

# -------------------- Utility: Synthetic dataset generator --------------------
@st.cache_data
def generate_synthetic_profiles(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    names = [
        "John", "Alice", "Bob", "Priya", "Rahul", "Sophia", "Michael", "Aisha",
        "Arturo", "Li", "Chen", "Olga", "Fatima", "Daniel", "Emma", "Noah"
    ]
    bio_templates_real = [
        "Photographer | Travel lover | Based in {city} | Contact: {email}",
        "Software Engineer @company | Loves open source and coffee",
        "Mom, runner, former teacher. Sharing recipes and family moments.",
        "Researcher in AI. Papers: {paper}",
    ]
    bio_templates_fake = [
        "Earn $500/day working from home! Click {link}",
        "Win a free iPhone now: {link}",
        "DM for followers boost and likes. Limited spots! {link}",
        "Cheap designer bags — wholesale prices {link}",
    ]
    cities = ["Mumbai", "Bengaluru", "New York", "London", "Delhi", "Pune", "Chennai", "Hyderabad"]

    records = []
    for i in range(n):
        is_fake = rng.rand() < 0.35  # 35% fake in synthetic set
        name = rng.choice(names) + (" " + rng.choice(names) if rng.rand() < 0.2 else "")
        username = (name.split()[0].lower() + str(rng.randint(1,9999)))
        if not is_fake:
            bio = rng.choice(bio_templates_real).format(city=rng.choice(cities), email=f"{username}@mail.com", paper=f"Paper{rng.randint(1,50)}")
            followers = int(np.clip(rng.normal(1500, 2500), 20, 200000))
            following = int(np.clip(rng.normal(400, 800), 10, 50000))
            posts = int(np.clip(rng.normal(300, 400), 0, 50000))
            has_profile_pic = 1
            account_age_days = int(np.clip(rng.normal(1200, 900), 10, 7000))
            external_links = rng.randint(0,2)
        else:
            bio = rng.choice(bio_templates_fake).format(link=f"http://buy{rng.randint(1,999)}.com")
            followers = int(np.clip(rng.normal(120, 300), 0, 50000))
            following = int(np.clip(rng.normal(2000, 4000), 0, 100000))
            posts = int(np.clip(rng.normal(30, 60), 0, 1000))
            has_profile_pic = int(rng.rand() < 0.6)  # some fakes have generic pics
            account_age_days = int(np.clip(rng.normal(40, 180), 0, 2000))
            external_links = rng.randint(1,5)

        records.append({
            "name": name,
            "username": username,
            "bio": bio,
            "followers": followers,
            "following": following,
            "posts": posts,
            "has_profile_pic": has_profile_pic,
            "account_age_days": account_age_days,
            "external_links": external_links,
            "label": int(is_fake),
        })
    return pd.DataFrame.from_records(records)


# -------------------- ML pipeline building --------------------
@st.cache_data
def build_and_train_model(df):
    # Text: combine name + username + bio
    df = df.copy()
    df['text'] = (df['name'].fillna('') + ' ' + df['username'].fillna('') + ' ' + df['bio'].fillna(''))

    X_text = df['text']
    X_num = df[['followers','following','posts','has_profile_pic','account_age_days','external_links']]
    y = df['label']

    text_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1,2), stop_words='english'))
    ])
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])

    pre = ColumnTransformer([
        ('text', text_pipe, 'text'),
        ('num', num_pipe, ['followers','following','posts','has_profile_pic','account_age_days','external_links'])
    ])

    clf = Pipeline([
        ('pre', pre),
        ('rf', RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    X = df[['text','followers','following','posts','has_profile_pic','account_age_days','external_links']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, preds)
    try:
        roc = roc_auc_score(y_test, probs)
    except Exception:
        roc = float('nan')

    report = classification_report(y_test, preds, output_dict=True)
    return clf, acc, roc, report, X_test, y_test, probs


# -------------------- Streamlit UI --------------------
st.title("🕵️ Fake Profile Detection — Demo")
st.markdown("A simple demo of a machine-learning pipeline that predicts whether a social profile looks fake. This is a teaching/demo tool, not a production fraud detector.")

col1, col2 = st.columns([1,2])
with col1:
    st.sidebar.header("Options")
    mode = st.sidebar.selectbox("Data source / Mode", ['Use demo synthetic dataset', 'Upload CSV (custom)'])
    n = st.sidebar.slider("Synthetic dataset size", 200, 5000, 2000, step=200)
    retrain = st.sidebar.button("Retrain model")
    st.sidebar.markdown("\n**CSV format if uploading:** columns required: name, username, bio, followers, following, posts, has_profile_pic (0/1), account_age_days, external_links, label (optional)")

with col2:
    st.header("Model & Data")

# Load data
if mode == 'Use demo synthetic dataset':
    df = generate_synthetic_profiles(n=n)
    st.success(f"Loaded synthetic dataset with {len(df)} rows")
else:
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded uploaded CSV with {len(df)} rows")
    else:
        st.info("Please upload a CSV to proceed or switch to the demo dataset mode.")
        st.stop()

# Train or load model (training live — fine for demo small datasets)
model_key = f"fake_profile_model_{len(df)}"
if 'model_store' not in st.session_state:
    st.session_state['model_store'] = {}

if (model_key not in st.session_state['model_store']) or retrain:
    with st.spinner('Training model...'):
        clf, acc, roc, report, X_test, y_test, probs = build_and_train_model(df)
        st.session_state['model_store'][model_key] = {
            'clf': clf,
            'acc': acc,
            'roc': roc,
            'report': report,
        }
else:
    data = st.session_state['model_store'][model_key]
    clf = data['clf']
    acc = data['acc']
    roc = data['roc']
    report = data['report']

st.subheader("Model performance on held-out test set (synthetic)")
st.metric("Accuracy", f"{acc:.3f}")
st.metric("ROC AUC", f"{roc:.3f}" if not np.isnan(roc) else "n/a")

st.write(pd.DataFrame(report).transpose())

# Show sample data
st.subheader("Sample profiles")
st.dataframe(df.sample(min(10, len(df))).reset_index(drop=True))

# Prediction form
st.subheader("Try the detector — input a profile")
with st.form('predict_form'):
    name_in = st.text_input('Name', 'Ajay Kumar')
    username_in = st.text_input('Username', 'ajay123')
    bio_in = st.text_area('Bio / Description', 'Photographer. Travel. Contact: ajay@mail.com')
    followers_in = st.number_input('Followers', min_value=0, value=1200)
    following_in = st.number_input('Following', min_value=0, value=450)
    posts_in = st.number_input('Posts', min_value=0, value=320)
    has_pic_in = st.selectbox('Has profile pic?', [1,0], format_func=lambda x: 'Yes' if x==1 else 'No')
    age_in = st.number_input('Account age (days)', min_value=0, value=800)
    links_in = st.number_input('External links count', min_value=0, value=1)
    submitted = st.form_submit_button('Check profile')

if submitted:
    input_text = {'text': [f"{name_in} {username_in} {bio_in}"],
                  'followers':[followers_in],
                  'following':[following_in],
                  'posts':[posts_in],
                  'has_profile_pic':[has_pic_in],
                  'account_age_days':[age_in],
                  'external_links':[links_in]}
    X_input = pd.DataFrame.from_dict(input_text)
    pred = clf.predict(X_input)[0]
    prob = clf.predict_proba(X_input)[0][1]

    if pred == 1:
        st.error(f"Model says: LIKELY FAKE (probability {prob:.2f})")
    else:
        st.success(f"Model says: LIKELY REAL (fake probability {prob:.2f})")

    # Show explainability: top contributing tf-idf tokens
    try:
        # access tfidf vectorizer from pipeline
        vect = clf.named_steps['pre'].named_transformers_['text'].named_steps['tfidf']
        rf = clf.named_steps['rf']
        # get feature importances for numeric features vs text features is non-trivial; approximate using permutation on text tokens
        st.subheader('Quick feature hints (approx)')
        # Show top TF-IDF tokens from the input
        tokens = vect.build_analyzer()(X_input['text'].iloc[0])
        top_tokens = sorted(set(tokens), key=lambda t: -len(t))[:10]
        st.write('Tokens found in input (sample):', top_tokens[:20])

        # Show numeric feature values
        st.write('Numeric features:', {k: v[0] for k, v in input_text.items() if k!='text'})
    except Exception as e:
        st.write('Could not produce token hints:', e)

# Allow user to download model (joblib)
st.sidebar.markdown('---')
if st.sidebar.button('Download trained model (.joblib)'):
    buffer = io.BytesIO()
    joblib.dump(clf, buffer)
    buffer.seek(0)
    st.sidebar.download_button('Download model file', data=buffer, file_name='fake_profile_model.joblib')

# Allow CSV download of synthetic dataset
if st.sidebar.button('Download sample CSV (synthetic)'):
    towrite = io.BytesIO()
    df.to_csv(towrite, index=False)
    towrite.seek(0)
    st.sidebar.download_button('Download CSV', data=towrite, file_name='synthetic_profiles.csv')

st.sidebar.markdown('\n---\nNotes:')
st.sidebar.write('• This demo uses a synthetic dataset and a simple RandomForest; it is only for education and prototyping.\n• For production: gather trustworthy labeled data, implement robust feature engineering (image analysis, temporal signals, network features), fairness checks, and human-in-the-loop review.')


# EOF
