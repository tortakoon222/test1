import json
import time
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


st.header("การวิเคราะห์ความรู้สึกภาษาไทย")
st.subheader("Tortakoon Sukpan")
st.subheader("NPRU ratchabhat nakhon pathom university")
st.subheade("DS Data Science")

col1, col2 = st.columns(2)
with col1:
    st.image('./pic/gg.jpg')
    lot3="https://lottie.host/340c4688-7333-4aa7-960b-cb7d695029a3/2maxe4TtSv.json"
    lottie3 = load_lottieurl(lot3)
    st_lottie(lottie3)
with col2:
    st.image('./pic/DS1.jpg')
st.balloons()
