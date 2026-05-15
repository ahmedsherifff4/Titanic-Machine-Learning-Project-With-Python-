import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import ProjectAI_py_


model = ProjectAI_py_.best_model
train = ProjectAI_py_.train



st.set_page_config(
    page_title="Titanic AI Project",
    layout="wide"
)

#####
st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #262730;
}

/* Buttons */
.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

/* Input Boxes */
.stTextInput>div>div>input {
    background-color: #1E1E1E;
    color: white;
}

.stNumberInput input {
    background-color: #1E1E1E;
    color: white;
}

/* Selectbox */
.stSelectbox div[data-baseweb="select"] {
    background-color: #1E1E1E;
    color: white;
}

</style>
""", unsafe_allow_html=True)


#####

st.title("🚢 Titanic AI Full Project")



menu = st.sidebar.selectbox(
    "Choose Section",
    ["Prediction",  "Dataset"]
)


# Dataset Section

if menu == "Dataset":

    st.header("📊 Dataset")

    dataset_choice = st.selectbox(
        "Choose Dataset",
        ["Train", "Test"]
    )
    num_rows = st.selectbox(
        "Choose Number of Rows",
        [10, 25, 50, 100, 200, 500]
    )



    if dataset_choice == "Train":
        data = ProjectAI_py_.train
    else:
        data = ProjectAI_py_.test

    st.dataframe(data.head(num_rows))
    st.write("Shape:", data.shape)


# Analysis Section
# Analysis Section




# Prediction Section


if menu == "Prediction":

    st.header("🔮 Passenger Survival Prediction")
    model_name = type(model).__name__
    st.write(f"🤖 Best Model Used :  {model_name}")
    passenger_id = st.text_input("Passenger ID ")
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    name = st.text_input("Name (e.g. Braund, Mr. Owen Harris)")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.number_input("Age", 0, 100, 25)
    sibsp = st.number_input("SibSp", 0, 10, 0)
    parch = st.number_input("Parch", 0, 10, 0)
    ticket = st.text_input("Ticket ")
    fare = st.number_input("Fare", 0.0, 600.0, 50.0)
    cabin = st.text_input("Cabin ")
    embarked = st.selectbox("Embarked", ["S", "C", "Q"])


    # Feature Engineering (SAME AS TRAINING)


    # Family size
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0

    # Title extraction
    title_match = pd.Series([name]).str.extract(r',\s*([^\.]+)\.')[0][0]

    common_titles = ['Mr', 'Miss', 'Mrs', 'Master']

    if pd.isna(title_match):
        title = "Rare"
    else:
        title = title_match.strip()
        if title not in common_titles:
            title = "Rare"


    age_bin = pd.cut(
        [age],
        bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Young', 'Adult', 'Senior']
    )[0]


    # Build input dataframe (RAW features)


    input_data = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Fare": [fare],
        "FamilySize": [family_size],
        "IsAlone": [is_alone],
        "Embarked": [embarked],
        "Title": [title],
        "AgeBin": [age_bin]
    })


    # Apply SAME preprocessing as training


    input_data = pd.get_dummies(input_data)

    # Align columns with training
    input_data = input_data.reindex(columns=ProjectAI_py_.X_train.columns, fill_value=0)

    # ======================================
    # Prediction
    # ======================================

    if st.button("Predict"):

        result = model.predict(input_data)

        if result[0] == 1:
            st.success("🎉 Passenger Survived")
        else:
            st.error("💀 Passenger Did Not Survive")
