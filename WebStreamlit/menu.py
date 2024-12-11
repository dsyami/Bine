import streamlit as st 
   
def menu():  
    st.sidebar.page_link("Home.py", label="🏠首页")  
    st.sidebar.page_link("pages/DataAnalysis.py", label="📊数据分析") 
    st.sidebar.page_link("pages/Model.py", label="✨模型结构")  
    st.sidebar.page_link("pages/Inference.py", label="🔍模型推理")  
