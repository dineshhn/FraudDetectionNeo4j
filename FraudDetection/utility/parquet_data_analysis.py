# import glob
# import json
# import os
# import streamlit as st
# from neo4j import GraphDatabase
# import altair as alt
# import pandas as pd
# import duckdb
# from openai import OpenAI
# from dotenv import load_dotenv
# from utility.common import get_base64_image
#
# def duck_parquet():
#     #Parquet folder path
#     DATA_PATH = r"C:\Users\Dinesh Narayana\Downloads\Fraud Detection Project\output_data"
#     TABLE_NAME = "transactions"
#
#     con = duckdb.connect(database=':memory:')
#     parquet_files = glob.glob(os.path.join(DATA_PATH, "*.parquet"))
#
#     if not parquet_files:
#         st.warning("No Parquet files found in the data directory.")
#     else:
#         con.execute(f"""
#         CREATE VIEW {TABLE_NAME} AS
#         SELECT * FROM read_parquet('{DATA_PATH}/*.parquet')
#         """)
#
#     st.subheader("💰 Total Transaction Amount per Type")
#     amount_by_type = con.execute("""
#         SELECT type, SUM(amount) AS total_amount
#         FROM transactions
#         GROUP BY type
#         ORDER BY total_amount DESC
#         """).df()
#
#     chart = alt.Chart(amount_by_type.reset_index()).mark_bar().encode(
#         x='type',
#         y='total_amount',
#         color='type'
#         ).properties(width=600, height=400)
#
#     st.altair_chart(chart, use_container_width=True)
#     #st.bar_chart(amount_by_type.set_index("type"))
#     #If normal bar chart is fine, remove altair chart
#     st.divider()
#
#     # Fraud count
#     st.subheader("🚨 Fraudulent Transactions Count")
#     fraud_count = con.execute("SELECT COUNT(*) FROM transactions WHERE isFraud = 1").fetchone()[0]
#     st.metric("Total Frauds", fraud_count)
#     st.divider()
#
#     # Avg amount by type
#     st.subheader("📊 Average Transaction Amount by Type")
#     avg_amount = con.execute("""
#         SELECT type, ROUND(AVG(amount), 2) AS avg_amount
#         FROM transactions
#         GROUP BY type
#         """).df()
#     s_df = avg_amount.style.set_properties(**{'text-align': 'left', 'background-color': '#f9f9f9',
#                                           'color': '#338', 'font-size': '14px', 'border': '1px solid #ddd'})
#     st.dataframe(s_df)
#     st.divider()
#
#     # Fraud types
#     st.subheader("📌 Fraud Types Count")
#     fraud_types = con.execute("""
#         SELECT type, COUNT(*) AS count
#         FROM transactions
#         WHERE isFraud = 1
#         GROUP BY type
#         """).df()
#     chart = alt.Chart(fraud_types).mark_bar().encode(
#         x='type',
#         y='count',
#         color='type',
#         tooltip=['count', 'type']
#         ).properties(
#             width=600,
#             height=400
#             #title="Transaction Amount by Account"
#             )
#     st.altair_chart(chart, use_container_width=True)
#         #st.bar_chart(fraud_types.set_index("type"), use_container_width=True)
#     st.divider()
#
#     # Top fraud customers
#     st.subheader("🧑‍💻 Top Fraudulent Customers")
#     top_customers = con.execute("""
#         SELECT nameOrig, COUNT(*) as frauds
#         FROM transactions
#         WHERE isFraud = 1
#         GROUP BY nameOrig
#         ORDER BY frauds DESC
#         LIMIT 5
#         """).df()
#     st.table(top_customers)
#     st.divider()
#
# # st.subheader("🔍 Old Balance vs Amount (Fraud Only)")
# # scatter_df = con.execute("""
# # SELECT oldbalanceOrg, amount
# # FROM transactions
# # WHERE isFraud = 1
# # LIMIT 200
# # """).df()
#
# # scatter = alt.Chart(scatter_df).mark_circle(size=100).encode(
# #     x='oldbalanceOrg',
# #     y='amount',
# #     color='oldbalanceOrg',
# #
# #     tooltip=['amount', 'oldbalanceOrg']
# # ).properties(
# #     width=600,
# #     height=400
# #     #title='Transaction Amount vs Old Balance'
# # ).interactive()  # enable zoom & pan
#
# #st.altair_chart(scatter, use_container_width=True)
# # st.scatter_chart(scatter_df, use_container_width=True)
# # st.divider()
#
# # st.subheader("📉 Balance Change after Transaction")
# # balance_change = con.execute("""
# # SELECT oldbalanceOrg - newbalanceOrig AS balance_change
# # FROM transactions
# # LIMIT 100
# # """).df()
# # st.line_chart(balance_change)
# # st.divider()
