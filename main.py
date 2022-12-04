import streamlit as st
import pickle
import pandas as pd

header = st.container()
dataset = st.container()
predictions = st.container()

@st.cache
def get_model():
    regressor = pickle.load(open('./model_artifacts/linreg_pipe.pkl','rb'))
    
    return regressor

with header:
    st.title('Optimizing Health Insurance Pricing with Linear Regression')
    st.caption(
        'Insurance companies can benefit from customer information in business decision making, such as pricing its service.\
        Having historical data to extract profile information on charges behavior might be useful for that purpose.'
    )



with dataset:
    st.header('About the Data')
    st.caption(
        'The chosen dataset contains medical costs of people characterized by certain attributes and is available at \
        [GitHub](https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv).' 
    )
    st.subheader('Attribute Information')
    st.markdown(
    """
    - **Age**: Age of primary beneficiary.
    - **Sex**: Insurance contractor gender (female, male).
    - **BMI**: Body mass index, providing an understanding of body (weights that are relatively high or low relative to height).
    - **Children**: Number of children covered by health insurance / Number of dependents.
    - **Smoker**: Indicates whether the beneficiary is a smoker.
    - **Region**: The beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
    - **Charges**: Individual medical costs billed by health insurance.
    """
    )


with predictions:
    st.header('Predicting Insurance Costs')

    input_col, result_col = st.columns(2)

    # input section
    input_col.markdown(
        """
        ### Insured Information
        """
    )
    input_col.caption(
        'Please provide the information below to estimate the cost of insurance.'
    )

    children = input_col.slider(
        "How many dependents does the beneficiary have?",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )

    age = input_col.number_input(
        "What is the beneficiary's age?",
        min_value=18,
        max_value=65,
        value=41,
        step=1
    )

    bmi = input_col.number_input(
        "Beneficiary's body mass index",
        min_value=15.0,
        max_value=55.0,
        value=35.0,
        step=0.05
    )


    smoker = input_col.selectbox(
        "Is the beneficiary a smoker?",
        options=['yes', 'no']
    )

    # output section

    result_col.markdown(
        """
        ### Predicted Values
        """
    )

    df = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker]
    })

    regressor = get_model()
    charges = round(regressor.predict(df)[0],2)

    result_col.caption("From the information provided:")

    result_col.write(f"The likely amount of charges associated with this profile is U$ {charges}.")