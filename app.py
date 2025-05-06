import streamlit as st
import pandas as pd
import joblib 
import numpy as np

st.set_page_config(page_title="HR Tools PT Jaya Jaya Maju", layout="wide")

# Load model Random Forest
@st.cache_resource
def load_rf_model():
    return joblib.load("attrition_model.pkl")  # Ubah ke nama file model kamu

# Preprocessing
def preprocess_data(df):
    df = df.copy()
    drop_cols = ['EmployeeId', 'Attrition', 'EmployeeCount', 'Over18', 'StandardHours']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    df = pd.get_dummies(df, columns=['BusinessTravel', 'Department', 'EducationField', 'Gender',
                                     'JobRole', 'MaritalStatus', 'OverTime'], drop_first=False)

    expected_features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'BusinessTravel_Travel_Frequently', 'BusinessTravel_Travel_Rarely',
        'Department_Research & Development', 'Department_Sales',
        'EducationField_Life Sciences', 'EducationField_Marketing',
        'EducationField_Medical', 'EducationField_Other', 'EducationField_Technical Degree',
        'Gender_Male', 'JobRole_Human Resources', 'JobRole_Laboratory Technician',
        'JobRole_Manager', 'JobRole_Manufacturing Director', 'JobRole_Research Director',
        'JobRole_Research Scientist', 'JobRole_Sales Executive', 'JobRole_Sales Representative',
        'MaritalStatus_Married', 'MaritalStatus_Single', 'OverTime_Yes'
    ]

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0 

    df = df[expected_features]

    return df

# Fungsi halaman prediksi
def predict_page():
    model = load_rf_model()

    st.markdown("<h1 style='text-align: center;'>📂 Prediksi Massal Resign Karyawan</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload file CSV berisi data karyawan", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Mentah yang Diupload")
        st.dataframe(raw_df.head())

        try:
            X_processed = preprocess_data(raw_df)

            preds = model.predict(X_processed)
            probs = model.predict_proba(X_processed)[:, 1]

            raw_df['Predicted_Attrition'] = preds
            raw_df['Probability'] = np.round(probs, 4)

            st.subheader("🎯 Hasil Prediksi")
            st.dataframe(raw_df[['EmployeeId', 'Predicted_Attrition', 'Probability']])

            csv = raw_df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Unduh Hasil Prediksi", data=csv, file_name='prediksi_attrition.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")

# Fungsi halaman utama
def home_page():
    st.title('👥 HR Tools - PT Jaya Jaya Maju')
    st.write('Selamat datang di homepage!')
    st.markdown("""
Aplikasi internal milik PT Jaya Jaya Maju yang dirancang khusus untuk membantu tim Human Resources dalam melakukan analisis data karyawan secara lebih efektif dan berbasis teknologi.

Dengan alat ini, Anda dapat melakukan prediksi terhadap kemungkinan seorang karyawan akan mengundurkan diri (attrition) berdasarkan berbagai faktor seperti demografi, performa kerja, kepuasan kerja, dan riwayat pekerjaan.  
Model prediktif yang digunakan didasarkan pada algoritma *machine learning* dan telah dilatih menggunakan data historis karyawan, sehingga mampu memberikan hasil yang cukup akurat untuk mendukung pengambilan keputusan strategis.

Gunakan menu di sisi kiri untuk mengunggah data karyawan dalam format CSV dan lihat hasil prediksi secara massal dalam hitungan detik.  
Prediksi ini dapat membantu Anda dalam menyusun strategi retensi karyawan, mengidentifikasi risiko turnover lebih awal, serta merancang program pengembangan SDM yang lebih terarah.

Terima kasih telah menggunakan HR Tools — mari kita bangun lingkungan kerja yang lebih baik dan berkelanjutan bersama-sama.
    """)

# Sidebar navigasi
def toggle_burger():
    pages = {
        'Home': home_page,
        'Prediksi Massal': predict_page,
    }
    st.sidebar.title("PT Jaya Jaya Maju")
    st.sidebar.image("logo2.png", width=200)
    page = st.sidebar.selectbox('Pilih Halaman', list(pages.keys()))
    pages[page]()

if __name__ == '__main__':
    toggle_burger()
