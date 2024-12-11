import streamlit as st
import pandas as pd
import time

def expensive_process(option, add):
    with st.spinner('Processing...'):
        time.sleep(5)
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C':[7, 8, 9]}) + add
    return (df, add)

cols = st.columns(2)
option = cols[0].selectbox('Select a number', options=['1', '2', '3'])
add = cols[1].number_input('Add a number', min_value=0, max_value=10)
if 'processed' not in st.session_state:
    st.session_state.processed = {}
if st.button('Process'):
    result = expensive_process(option, add)
    st.session_state.processed[option] = result
if option in st.session_state.processed:
    st.write(f'Option {option} processed with add {add}')
    st.write(st.session_state.processed[option][0])
