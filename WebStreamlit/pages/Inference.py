import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from urllib.error import URLError
from Controller.ModelController import ModelController
from draw.confusionMatrix import draw_confusion_matrix
from menu import menu


st.set_page_config(page_title="Mapping Demo", page_icon="🌍")
menu()
st.markdown("### EEG 运动想象分类")
st.sidebar.header("EEG 运动想象分类")
st.write(
    """基于 GConvLSTM 的EEG运动想象分类任务"""
)


@st.cache_data
def load_dataset(dir):
    all_set = np.load(dir + 'all_set.npy')
    all_label = np.load(dir + 'all_label.npy')
    return all_set, all_label

def extract_numbers_to_int_list(strings):
    import re
    return int(re.search(r'\d+$', strings).group())

try:
    sex_data = st.sidebar.selectbox(
        label = '请选择数据集',
        options = ('PhysioNet MI Dataset'),
        index=0,
        format_func=str,
    )

    models = ['GConvLSTM']
    sex_model = st.sidebar.selectbox(
        label = '请选择模型',
        options = models,
        format_func = str,
        help = '选择推理使用的模型'
    )

    subjects = ['all', '20Subjects']
    for i in range(1, 109):
        subjects.append(f"Subject {i}")

    mts_subject = st.sidebar.multiselect(
        label = '请选择被试数据集',
        options = subjects,
        format_func = str,
        help = '根据被试选择用于推理的数据集'
        )
    
    adj_methods = ['Pearson', 'xcorr', 'PLV']
    sex_adj = st.sidebar.selectbox(
        label = '请选择临界矩阵计算方法',
        options = adj_methods,
        format_func = str,
        help = '根据选择的方法构建脑网络图'
    )

    button_status = st.sidebar.button(label='显示结果')

    if not mts_subject:
        st.write("请选择被试构造推理数据集")
    elif mts_subject and not button_status:
        st.write("选择成功，点击按钮进行推理")

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    if button_status:
        if mts_subject == None:
            st.error("请选择被试数据集")
            st.stop()

        dir = 'D:/dsHit/thesis/MyProject/PreprecessedDataset/For-GCN-based-Models/PhysioNet-Dataset/outputdata_108-subjects/subjects/'
        modelController = ModelController(sex_model, sex_adj)
        set, label = load_dataset(dir)
        label_name = ['L', 'R', 'B', 'F']
        if 'all' in mts_subject:
            index_subject = [i for i in range(1, 109)]
        elif '20Subjects' in mts_subject:
            index_subject = [i for i in range(1, 21)]
        else:
            index_subject = [extract_numbers_to_int_list(i) for i in mts_subject]

        loss_list = []
        eval_list = []
        pred_list = []
        target_list = []
        for i, index in enumerate(index_subject):
            status_text.text("完成%i%%" % (int(i + 1) / len(index_subject) * 100))
            progress_bar.progress(int(i + 1) / len(index_subject))

            loss, eval, pred, target = modelController.predict(set[index - 1], label[index - 1])
            eval['subject'] = f'subject_{index}'
            loss_list.append(loss)
            eval_list.append(eval)
            pred_list.append(pred)
            target_list.append(target)
        progress_bar.empty()

        tab1, tab2 = st.tabs(["🗃 数据", "📈 图表"])
        tab1.subheader("EEG MI Inference 结果")
        for i, subject in enumerate(index_subject):
            tab1.write(f"### subject {subject}")
            tab1.write(f"loss: {loss_list[i]}")
            tab1.write(eval_list[i])
            buf = draw_confusion_matrix(target_list[i], pred_list[i], label_name=label_name)
            tab1.image(buf, use_container_width=True)
        
        df = pd.DataFrame(eval_list)
        # 提取subject中的数字部分，并转换为整数，然后基于这个数字进行排序
        # 这里使用str.extract和正则表达式来提取数字
        df['subject_number'] = df['subject'].str.extract(r'subject_(\d+)').astype(int)
        # key参数指定排序的规则，而不是默认的字符串排序
        df_sorted = df.sort_values(by='subject_number', key=lambda x: pd.to_numeric(x, errors='coerce')).drop(columns='subject_number')
        subject_num = df_sorted['subject'].tolist()
        # 设置Streamlit页面布局
        st.title('Evaluation Metrics for Subjects')
        
        # 设置颜色映射
        color_map = {
            'accuracy': '#e7ba52',
            'recall': '#a7a7a7',
            'precision': '#aec7e8',
            'f1_score': '#1f77b4',
            'kappa': '#9467bd'
        }

        # 使用Altair创建交互式柱状图函数
        def create_bar_chart(df, metric):
            # 获取颜色
            color = color_map.get(metric, 'gray')  # 如果没有找到匹配的颜色，则使用灰色
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('subject:N', sort=alt.Sort(subject_num)),  # X轴为subject
                y=f'{metric}:Q',  # Y轴为指定的度量指标
                color=alt.value(color),
                tooltip=[alt.Tooltip('subject'), alt.Tooltip(metric)]  # 添加提示信息
            ).properties(
                title=f'{metric.capitalize()} for Each Subject',  # 图表标题
                width=600,  # 图表宽度
                height=400  # 图表高度
            )
            return chart
        
        # 为每个指标创建交互式柱状图并展示
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'kappa']
        for metric in metrics:
            chart = create_bar_chart(df_sorted, metric)
            # 使用st.altair_chart展示Altair图表
            tab2.altair_chart(chart, use_container_width=True)

        # for metric in metrics:
        #     values = [eval_list[i][metric] for i, subject in enumerate(index_subject)]
        #     tab2.subheader(metric.capitalize())
        #     fig, ax = plt.subplots(figsize=(10, 6))
        #     ax.bar(labels, values)
        #     ax.set_xlabel = ('Subjects')
        #     ax.set_ylabel = (metric.capitalize())
        #     ax.set_title(f'{metric.capitalize()} for Each Subject')
        #     tab2.pyplot(fig)
    

except URLError as e:
    st.error(
        """
        **This demo requires internet access.**
        Connection error: %s
    """
        % e.reason
    )
