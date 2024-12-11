import streamlit as st
import numpy as np
from urllib.error import URLError
from Controller.SubjectController import SubjectController
from menu import menu


st.set_page_config(page_title="数据分析", page_icon="📊")

menu()
st.markdown("### 脑网络邻接矩阵可视化")
st.sidebar.header("数据分析")

@st.cache_data
def load_dataset(dir):
    all_set = np.load(dir + 'all_set.npy')
    all_label = np.load(dir + 'all_label.npy')
    return all_set, all_label

try:
    sex_data = st.sidebar.selectbox(
        label = '请选择数据集',
        options = ('PhysioNet MI Dataset', 'HighGamma Dataset'),
        index=0,
        format_func=str,
    )

    if sex_data == 'HighGamma Dataset':
        subjects = [f"Subject {i}" for i in range(1, 10)]
    elif sex_data == 'PhysioNet MI Dataset':
        subjects = [f"Subject {i}" for i in range(1, 109)]
    sex_subject = st.sidebar.selectbox(
        label = '请选择被试',
        options = subjects,
        format_func = str,
        help = '选择被试以生成脑网络邻接矩阵热力图'
        )
    

    adj_methods = ['Pearson', 'xcorr', 'PLV']
    sex_adj = st.sidebar.selectbox(
        label = '请选择邻接矩阵计算方法',
        options = adj_methods,
        format_func = str,
        help = ''
    )

    button_status = st.sidebar.button(label='显示结果')

    if not button_status:
        st.write('点击按钮进行结果展示')

    if button_status:
        dir = 'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_108-subjects/subjects/'
        subjectController = SubjectController()
        set, label = load_dataset(dir)
        figs = subjectController.draw_adj_by_subjects(set, label, sex_subject)
        for i, f in enumerate(figs):
            st.write(f)
    

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
