import streamlit as st
from PIL import Image
from menu import menu

st.set_page_config(page_title="模型结构", page_icon="📈")
menu()
st.markdown("### 模型结构")
st.sidebar.header("模型结构")
st.write(
    """本项目的整体结构设计"""
)

image_url = './asset/project.png'
image = Image.open(image_url)
# 使用st.image函数展示图像
st.image(image, caption='项目整体结构图')