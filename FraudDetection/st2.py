import glob
import json
import os
import streamlit as st
from neo4j import GraphDatabase
import altair as alt
import pandas as pd
import duckdb
from openai import OpenAI
from dotenv import load_dotenv
from utility.common import get_base64_image

# from neo4j_viz import GraphViz

st.set_page_config(layout="wide", page_title="Fin-Fraud Detection", page_icon=":anchor:")

# #If background enabled, uncomment below
# image_path = r"C:\Users\Dinesh Narayana\Downloads\FraudDetection\utility\images1.jpg"
# base64_image = get_base64_image(image_path)

#Add this to st.markdown
# .stApp {{
#     background-image: url("data:image/jpg;base64,{base64_image}");
# background-size: cover;
# background-repeat: no-repeat;
# background-attachment: fixed;
# height: 100vh;
# width: 100vw;
# }}


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

st.title("📊 Financial Fraud Detection")

load_dotenv()  # load environment variables
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USER')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

#Parquet folder path
DATA_PATH = r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\output_data"
TABLE_NAME = "transactions"

con = duckdb.connect(database=':memory:')
parquet_files = glob.glob(os.path.join(DATA_PATH, "*.parquet"))

if not parquet_files:
    st.warning("No Parquet files found in the data directory.")
else:
    con.execute(f"""
        CREATE VIEW {TABLE_NAME} AS 
        SELECT * FROM read_parquet('{DATA_PATH}/*.parquet')
    """)

# Create intuitive tab layout
#Add pages as needed
tab1, tab2, tab3 = st.tabs(["📊 Parquet Data Analysis", "🧠 Neo4j Data Analysis", "🤖 LLM Query"])

# Page 1
with tab1:
    # st.title("Transactional Data Analysis")

    st.subheader("💰 Total Transaction Amount per Type")
    amount_by_type = con.execute("""
    SELECT type, SUM(amount) AS total_amount
    FROM transactions
    GROUP BY type
    ORDER BY total_amount DESC
    """).df()

    chart = alt.Chart(amount_by_type.reset_index()).mark_bar().encode(
        x='type',
        y='total_amount',
        color='type'
    ).properties(width=600, height=400)

    st.altair_chart(chart, use_container_width=True)
    #st.bar_chart(amount_by_type.set_index("type"))
    #If normal bar chart is fine, remove altair chart
    st.divider()

    # Fraud count
    st.subheader("🚨 Fraudulent Transactions Count")
    fraud_count = con.execute("SELECT COUNT(*) FROM transactions WHERE isFraud = 1").fetchone()[0]
    st.metric("Total Frauds", fraud_count)
    st.divider()

    # Avg amount by type
    st.subheader("📊 Average Transaction Amount by Type")
    avg_amount = con.execute("""
    SELECT type, ROUND(AVG(amount), 2) AS avg_amount
    FROM transactions
    GROUP BY type
    """).df()
    s_df = avg_amount.style.set_properties(**{'text-align': 'left', 'background-color': '#f9f9f9',
                                               'color': '#338', 'font-size': '14px', 'border': '1px solid #ddd'})
    st.dataframe(s_df)
    st.divider()

    # Fraud types
    st.subheader("📌 Fraud Types Count")
    fraud_types = con.execute("""
    SELECT type, COUNT(*) AS count
    FROM transactions
    WHERE isFraud = 1
    GROUP BY type
    """).df()

    chart = alt.Chart(fraud_types).mark_bar().encode(
        x='type',
        y='count',
        color='type',
        tooltip=['count', 'type']
    ).properties(
        width=600,
        height=400
        #title="Transaction Amount by Account"
    )
    st.altair_chart(chart, use_container_width=True)
    #st.bar_chart(fraud_types.set_index("type"), use_container_width=True)
    st.divider()

    # Top fraud customers
    st.subheader("🧑‍💻 Top Fraudulent Customers")
    top_customers = con.execute("""
    SELECT nameOrig, COUNT(*) as frauds
    FROM transactions
    WHERE isFraud = 1
    GROUP BY nameOrig
    ORDER BY frauds DESC
    LIMIT 5
    """).df()
    st.table(top_customers)
    st.divider()

    # st.subheader("🔍 Old Balance vs Amount (Fraud Only)")
    # scatter_df = con.execute("""
    # SELECT oldbalanceOrg, amount
    # FROM transactions
    # WHERE isFraud = 1
    # LIMIT 200
    # """).df()

    # scatter = alt.Chart(scatter_df).mark_circle(size=100).encode(
    #     x='oldbalanceOrg',
    #     y='amount',
    #     color='oldbalanceOrg',
    #
    #     tooltip=['amount', 'oldbalanceOrg']
    # ).properties(
    #     width=600,
    #     height=400
    #     #title='Transaction Amount vs Old Balance'
    # ).interactive()  # enable zoom & pan

    #st.altair_chart(scatter, use_container_width=True)
    # st.scatter_chart(scatter_df, use_container_width=True)
    # st.divider()

    # st.subheader("📉 Balance Change after Transaction")
    # balance_change = con.execute("""
    # SELECT oldbalanceOrg - newbalanceOrig AS balance_change
    # FROM transactions
    # LIMIT 100
    # """).df()
    # st.line_chart(balance_change)
    # st.divider()

with tab2:
    # st.title("🔍 Neo4j Graph Analysis")
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Reusable query executor
    def run_query(tx, query):
        return [dict(record) for record in tx.run(query)]

    g_query = {
        "Total Number of Accounts": """
        MATCH (a:Account) RETURN count(a) AS total_accounts;
    """,
        "Total Number Transactions": """
        MATCH ()-[r:TRANSACTION]->() RETURN count(r) AS total_transactions;
    """,
        "Sample Transactions (Limit 100)": """
        MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
        RETURN a1.id AS Sender, a2.id AS Receiver,round(t.amount, 2) as Amount, t.type as Type, t.isFraud as isFraud
        LIMIT 100
    """,
        "Fraudulent Transactions": """
        MATCH (a1:Account)-[t:TRANSACTION {isFraud: true}]->(a2:Account)
        RETURN a1.id AS Fraudster, a2.id AS Victim, round(t.amount, 2) as Amount, t.type as Type
    """,
        "Transaction Type Fraud Count": """
        MATCH (:Account)-[t:TRANSACTION]->(:Account)
        WHERE t.isFraud = true
        RETURN t.type AS TransactionType, COUNT(*) AS FraudCount;
    """,
        "Accounts That Sent Fraudulent Transfers": """
        MATCH (a:Account)-[t:TRANSACTION {isFraud: true}]->()
        RETURN DISTINCT a.id AS Fraudster, count(t) AS fraud_count
        ORDER BY fraud_count DESC;
    """,
        "Top Transaction Amounts": """
        MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
        RETURN a1.id AS Sender, a2.id AS Receiver, round(t.amount, 2) as Amount, t.isFraud as isFraud
        ORDER BY t.amount DESC
        LIMIT 10;
    """,
        "Top 5 Accounts with high number of transactions": """
        MATCH (a:Account)-[t:TRANSACTION]->()
        RETURN a.id AS Account, COUNT(*) AS tx_count
        ORDER BY tx_count DESC
        LIMIT 5
    """,
        "Account Pairs with Multiple Transactions Between Them": """
        MATCH (a1:Account)-[t:TRANSACTION]->(a2:Account)
        WITH a1, a2, count(*) AS tx_count
        WHERE tx_count > 1
        RETURN a1.id AS Sender, a2.id AS Receiver, tx_count
        ORDER BY tx_count DESC;
    """,
        "Count of Fraud by Step": """
        MATCH ()-[t:TRANSACTION {isFraud: true}]->()
        RETURN t.step AS TimeStep, COUNT(*) AS FraudCount
        ORDER BY TimeStep
    """,
    }

    selection = st.selectbox("Choose a query", list(g_query.keys()))

    with driver.session() as session:
        #with st.status("Processing...", expanded=True) as status:
        with st.spinner("Querying DB...", show_time=True):
            results = session.execute_read(run_query, g_query[selection])
            df = pd.DataFrame(results)

            st.subheader("🔎 Query Results")
            st.table(df.reset_index(drop=True))
            #status.update(label="Done!", state="complete", expanded=False)

            if selection == "Transaction Type Fraud Count":
                if df.empty:
                    print("Nothing to show")
                else:
                    st.bar_chart(df.set_index("type"))

    # driver.close()

# Page 2 – LLM Query
# Keep the docker desktop running, with mistral llm in this case
with tab3:
    st.title("Ask LLM")

    def llm_cypher_query(account_id: str):
        query = f"""
                MATCH (a:Account)-[t:TRANSACTION]->(b:Account)
                WHERE a.id = '{account_id}' OR b.id = '{account_id}'
                RETURN a.id AS Sender, b.id AS Receiver, round(t.amount, 2) AS Amount, t.type AS Type, 
                t.isFraud AS IsFraud, t.isFlaggedFraud AS IsFlaggedFraud
                LIMIT 50
                """
        with driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]

    user_input = st.text_input("Enter an Account ID", placeholder="C1305486145")

    if st.button("Submit") and user_input:
        with st.spinner("Fetching data from Neo4j DB...", show_time=True):
            cypher_op = llm_cypher_query(user_input.strip())
            df_llm = pd.DataFrame(cypher_op)
            st.subheader(f"🔎 Rows Returned for Account ID : {user_input}")
            st.table(df_llm)
            st.divider()

        client = OpenAI(
            base_url="http://localhost:12434/engines/llama.cpp/v1/",  # llama.cpp server
            api_key="llama"
            )

        try:
            if cypher_op:
                context_text = json.dumps(cypher_op, indent=2)
                with st.spinner("Waiting for LLM response...", show_time=True):
                    response = client.chat.completions.create(
                        model="ai/mistral",  # model name from docker
                        messages=[{"role": "user", "content": f"Based on the following transactions, explain any signs of fraud:\n{context_text}"}],
                        temperature=0.7,
                        max_tokens=256
                        )

                    st.markdown("### Response:")
                    st.write(response.choices[0].message.content)
            else:
                st.warning("No data found for that account.")
        except Exception as e:
            st.error(f"Error connecting to LLM: {e}")
