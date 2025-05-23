from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

st.title('การจำแนกข้อมูลด้วยเทคนิค K-Nearest Neighbor')

col1, col2 = st.columns(2)

with col1:
    st.header("พงศกร บุญสม")
    st.image("./img/heart1.jpg")

with col2:
    st.header("การทำนายโรคหัวใจ")
    st.image("./img/heart2.jpg")

html_7 = """
<div style="background-color:#39edf6;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h4>ข้อมูลโรคหัวใจสำหรับทำนาย</h5></center>
</div>
"""
st.markdown(html_7, unsafe_allow_html=True)
st.markdown("")
st.markdown("")

st.subheader("ข้อมูลส่วนแรก 10 แถว")
dt = pd.read_csv("./data/Heart3.csv")
st.write(dt.head(10))
st.subheader("ข้อมูลส่วนสุดท้าย 10 แถว")
st.write(dt.tail(10))

# สถิติพื้นฐาน
st.subheader("📈 สถิติพื้นฐานของข้อมูล")
st.write(dt.describe())

# การเลือกแสดงกราฟตามฟีเจอร์
st.subheader("📌 เลือกฟีเจอร์เพื่อดูการกระจายข้อมูล")
feature = st.selectbox("เลือกฟีเจอร์", dt.columns[:-1])

# วาดกราฟ boxplot
st.write(f"### 🎯 Boxplot: {feature} แยกตามชนิดของโรคหัวใจ")
fig, ax = plt.subplots()
sns.boxplot(data=dt, x='HeartDisease', y=feature, ax=ax)
st.pyplot(fig)

# วาด pairplot
if st.checkbox("แสดง Pairplot (ใช้เวลาประมวลผลเล็กน้อย)"):
    st.write("### 🌺 Pairplot: การกระจายของข้อมูลทั้งหมด")
    fig2 = sns.pairplot(dt, hue='HeartDisease')
    st.pyplot(fig2)

html_8 = """
<div style="background-color:#6BD5DA;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>ทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8, unsafe_allow_html=True)
st.markdown("")

sp_a = st.number_input("กรุณาเลือกข้อมูล Age")
sp_s = st.number_input("กรุณาเลือกข้อมูล Sex")
sp_c = st.number_input("กรุณาเลือกข้อมูล ChestPainType")
sp_r = st.number_input("กรุณาเลือกข้อมูล RestingBP")
sp_ctr = st.number_input("กรุณาเลือกข้อมูล Cholesterol")
sp_fbs = st.number_input("กรุณาเลือกข้อมูล FastingBS")
sp_recg = st.number_input("กรุณาเลือกข้อมูล RestingECG")
sp_m = st.number_input("กรุณาเลือกข้อมูล MaxHR")
sp_e = st.number_input("กรุณาเลือกข้อมูล ExerciseAngina")
sp_o = st.number_input("กรุณาเลือกข้อมูล Oldpeak")
sp_esl = st.number_input("กรุณาเลือกข้อมูล EST_Slope")

if st.button("ทำนายผล"):
    dt = pd.read_csv("./data/Heart3.csv") 
    X = dt.drop('HeartDisease', axis=1)
    y = dt.HeartDisease  

    Knn_model = KNeighborsClassifier(n_neighbors=3)
    Knn_model.fit(X, y)  
    
    x_input = np.array([[sp_a, sp_s, sp_c, sp_r, sp_ctr, sp_fbs, sp_recg, sp_m, sp_e, sp_o, sp_esl]])
    out = Knn_model.predict(x_input)
    
    st.write(out)

    if out[0] == 1:
        st.image("./img/heart2.jpg")
    else:
        st.image("./img/heart3.jpg")
else:
    st.write("ไม่ทำนาย")
