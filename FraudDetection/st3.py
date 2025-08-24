import os
import streamlit as st
from neo4j import GraphDatabase
from dotenv import load_dotenv
from utility.common import duck_db_parquet_analysis, gen_live_data , neo4j_analysis, llm_func


st.set_page_config(layout="wide", page_title="Fin-Fraud Detection", page_icon=":anchor:")
st.markdown(f"""
    <style>
 
    .stApp {{
        font-family: 'Segoe UI', sans-serif;
        padding: 1rem 2rem;
    }}

    /* Rounded widgets */
    .stButton>button {{
        border-radius: 10px;
        background-color: #b5fc03;
        color: black;
    }}

    .stTextInput>div>div>input {{
        border-radius: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📊 Financial Fraud Detection")

load_dotenv()  # load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Live Data creation. To check with ML alogithms.
with st.sidebar:
    st.title("Live Data")
    number_of_records = st.sidebar.number_input("Enter number of records (<5):", min_value=1, max_value=5)
    if st.sidebar.button("Submit"):
        data = gen_live_data(number_of_records).to_dict(orient="records")
        for i, tx in enumerate(data):
            st.write(f"**Transaction {i+1}**")
            st.write(f"- amount: {tx['amount']}")
            st.write(f"- oldbalanceOrg: {tx['oldbalanceOrg']}")
            st.write(f"- newbalanceOrig: {tx['newbalanceOrig']}")
            st.write(f"- oldbalanceDest: {tx['oldbalanceDest']}")
            st.write(f"- newbalanceDest: {tx['newbalanceDest']}")
            st.markdown("---")

#Create intuitive tab layout
#Add pages as needed
tab1, tab2, tab3 = st.tabs(["🦆 Duck DB Analysis", "🧠 Neo4j Data Analysis", "🤖 LLM Query"])

# Page 1
with tab1:
    duck_db_parquet_analysis()

with tab2:
    neo4j_analysis(driver)

with tab3:
    llm_func(driver)
    # LLM Query
    # Keep the docker desktop running, with mistral llm in this case