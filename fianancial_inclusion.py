import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings('ignore')



data = pd.read_csv('Financial_inclusion_dataset.csv')

st.markdown("<h1 style = 'color: #FF204E; text-align: center; font-size: 60px; font-family: Georgia'>BANK ACCOUNT PREDICTOR APP</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #A0153E; text-align: center; font-family: italic'>BUILT BY FRANCES </h4>", unsafe_allow_html = True)

st.markdown("<br>", unsafe_allow_html=True)



# #add image
st.image('pngwing.com (15).png',width = 600)

st.markdown("<h2 style = 'color: #132043; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)


st.markdown("Empowering individuals worldwide by providing access to essential financial services, fostering economic growth, and reducing inequalities through innovative technology and inclusive practices.Our objective is to promote financial inclusion by expanding access to banking, credit, insurance, and other financial services to underserved populations, thereby empowering individuals, fostering economic development, and reducing poverty.")
st.sidebar.image('pngwing.com (13).png',caption = 'Welcome User')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.divider()
st.header('Project Data')
st.dataframe(data, use_container_width = True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("<br>", unsafe_allow_html=True)



# primaryColor="#FF4B4B"  
# backgroundColor="#99E6FF"
# secondaryBackgroundColor="#CCFCFF"
# textColor="#331133"
# font="sans serif"


st.sidebar.subheader('User Input Variables')


sel_cols = ['age_of_respondent', 'household_size', 'job_type', 'education_level', 'marital_status', 'country',
            'location_type', 'relationship_with_head', 'bank_account']

age = st.sidebar.number_input('age_of_respondent', data['age_of_respondent'].min(), data['age_of_respondent'].max())
household = st.sidebar.number_input('household_size', data['household_size'].min(), data['household_size'].max())
job = st.sidebar.selectbox('job_type', data['job_type'].unique())
education = st.sidebar.selectbox('education_level', data['education_level'].unique())
marital = st.sidebar.selectbox('marital_status', data['marital_status'].unique())
country = st.sidebar.selectbox('country', data['country'].unique())
location = st.sidebar.selectbox('location_type', data['location_type'].unique())
rel_head = st.sidebar.selectbox('relationship_with_head', data['relationship_with_head'].unique())




#users input
input_var = pd.DataFrame()
input_var['age_of_respondent'] = [age]
input_var['household_size'] = [household]
input_var['job_type'] = [job]
input_var['education_level'] = [education]
input_var['marital_status'] = [marital]
input_var['country'] = [country]
input_var['location_type'] = [location]
input_var['relationship_with_head'] = [rel_head]


st.markdown("<br>", unsafe_allow_html= True)
st.divider()
st.subheader('Users Inputs')
st.dataframe(input_var, use_container_width = True)

# import the transformers
job_type = joblib.load('job_type_encoder.pkl')
education_level = joblib.load('education_level_encoder.pkl')
marital_status = joblib.load('marital_status_encoder.pkl')
country = joblib.load('country_encoder.pkl')
location_type = joblib.load('location_type_encoder.pkl')
relationship_with_head = joblib.load('relationship_with_head_encoder.pkl')
bank_account = joblib.load('bank_account_encoder.pkl')



# transform the users input with the imported encoders
input_var['job_type'] = job_type.transform(input_var[['job_type']])
input_var['education_level'] = education_level.transform(input_var[['education_level']])
input_var['marital_status'] = marital_status.transform(input_var[['marital_status']])
input_var['country'] = country.transform(input_var[['country']])
input_var['location_type'] = location_type.transform(input_var[['location_type']])
input_var['relationship_with_head'] = relationship_with_head.transform(input_var[['relationship_with_head']])




# st.header('Transformed Input Variable')
# st.dataframe(input_var, use_container_width = True)

# st.dataframe(input_var)
model = joblib.load('FinancialModell.pkl')





if st.button('Confirm your Eligibility'):
    predicted = model.predict(input_var)
    if predicted[0] == 0:
        st.error(f"Unfortunately...You are not eligibile to open or have a bank account")
        st.image('pngwing.com (7).png', width = 300) 

    else:
        st.success(f"Congratulations... You are eligibile to have and open a bank account.please proceed to any of our offices to open an account")
        st.image('pngwing.com (6).png', width = 300)
        st.balloons()
        
























