import streamlit as st
from menu import menu

st.set_page_config(
    page_title="你好",
    page_icon="👋",
)

st.write("# 欢迎使用  EEG-WebStreamlit！ 👋")


st.markdown(
    """
    EEG-WebStreamlit 是基于 Streamlit 框架作为前端的 EEG 运动想象脑机接口 Web 应用。
    - 基于 Pytorch2.0 和 PYG 框架实现基于图神经网络的深度学习模型。
    - 主要任务为运动想象分类任务
    - 主要数据集包括 [PhysioNet MI Dataset](https://physionet.org/content/eegmmidb/1.0.0/)

    **👈 从侧边栏选择一个功能**，看看 EEG-WebStreamlit 能做什么吧！
    ### 了解更多
    - 查看前端 Streamlit 框架 [文档](https://streamlit.io)
    - 查看模型 Pytorch 框架 [文档](https://pytorch.org/docs/stable/index.html)
    - 阅读项目源码 [文档](https://github.com/dsyami/Bine)
    ### 查看使用教程
    - [帮助文档](https://www.baidu.com/)
"""
)

def main():
    st.sidebar.title('首页')

if __name__ == "__main__":
    menu()
    main()