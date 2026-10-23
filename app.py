
import pickle
import pandas as pd
import streamlit as st
st.set_page_config(
    page_title="Job Placement Prediction",
    page_icon="🎯",
    layout="wide"
)
@st.cache_resource
def load_model():
    with open("job_placement_model.pkl", "rb") as file:
        return pickle.load(file)
model = load_model()
st.title("🎯 Job Placement Prediction System")
st.write(
    "Enter the candidate's details to predict their "
    "job placement outcome."
)
st.info(
    "This model estimates placement outcomes from historical "
    "data. Its predictions are not a guarantee of employment."
)
with st.form("candidate_form"):
    st.subheader("1. Personal and Educational Details")
    col1, col2 = st.columns(2)
    with col1:
        age_years = st.number_input(
            "Age (years)", 21, 34, 25
        )
        gender = st.selectbox(
            "Gender", ["female", "male"]
        )
        degree_specialization = st.selectbox(
            "Degree Specialization",
            [
                "computer science",
                "information technology",
                "electronics",
                "mechanical",
                "others"
            ]
        )
        ssc_percentage = st.number_input(
            "SSC Percentage", 50.0, 95.0, 75.0
        )
        hsc_percentage = st.number_input(
            "HSC Percentage", 50.0, 95.0, 75.0
        )
        degree_percentage = st.number_input(
            "Degree Percentage", 55.0, 95.0, 75.0
        )
    with col2:
        internship_experience = st.selectbox(
            "Internship Experience", ["yes", "no"]
        )
        career_switch_willingness = st.selectbox(
            "Career Switch Willingness",
            ["willing", "not willing", "unknown"]
        )
        relevant_experience = st.selectbox(
            "Relevant Experience",
            ["relevant", "not relevant", "unknown"]
        )
        years_of_experience = st.number_input(
            "Years of Experience", 0.0, 5.0, 1.0
        )
        certifications_count = st.number_input(
            "Number of Certifications", 0, 8, 1
        )
    st.subheader("2. Candidate Skills")

    col1, col2 = st.columns(2)
    with col1:
        technical_score = st.number_input(
            "Technical Score", 40.0, 100.0, 75.0
        )
        aptitude_score = st.number_input(
            "Aptitude Score", 40.0, 100.0, 70.0
        )
    with col2:
        communication_score = st.number_input(
            "Communication Score", 45.0, 100.0, 70.0
        )
        skills_match_percentage = st.number_input(
            "Skills Match (%)", 45.0, 100.0, 75.0
        )
    st.subheader("3. Job and Employment Details")
    col1, col2 = st.columns(2)
    with col1:
        company_tier = st.selectbox(
            "Company Tier", ["tier 1", "tier 2", "tier 3"]
        )
        job_role_match = st.selectbox(
            "Job Role Match",
            ["matched", "not matched", "unknown"]
        )
        competition_level = st.selectbox(
            "Competition Level", ["low", "medium", "high"]
        )
        bond_requirement = st.selectbox(
            "Bond Requirement", ["not required", "required"]
        )
        layoff_history = st.selectbox(
            "Previous Layoff History",
            ["no", "yes", "unknown"]
        )
    with col2:
        relocation_willingness = st.selectbox(
            "Relocation Willingness",
            ["willing", "not willing", "unknown"]
        )
        previous_ctc_lpa = st.number_input(
            "Previous CTC (LPA)", 0.0, 13.84, 4.0
        )
        expected_ctc_lpa = st.number_input(
            "Expected CTC (LPA)", 3.0, 26.52, 6.0
        )
        notice_period_days = st.number_input(
            "Notice Period (days)", 0, 90, 30
        )
        employment_gap_months = st.number_input(
            "Employment Gap (months)", 0, 18, 0
        )
    submitted = st.form_submit_button(
        "Predict Placement",
        type="primary",
        use_container_width=True
    )
if submitted:
    candidate = pd.DataFrame([{
        "age_years": age_years,
        "gender": gender,
        "degree_specialization": degree_specialization,
        "ssc_percentage": ssc_percentage,
        "hsc_percentage": hsc_percentage,
        "degree_percentage": degree_percentage,
        "internship_experience": internship_experience,
        "career_switch_willingness": career_switch_willingness,
        "relevant_experience": relevant_experience,
        "years_of_experience": years_of_experience,
        "certifications_count": certifications_count,
        "technical_score": technical_score,
        "aptitude_score": aptitude_score,
        "communication_score": communication_score,
        "skills_match_percentage": skills_match_percentage,
        "company_tier": company_tier,
        "job_role_match": job_role_match,
        "competition_level": competition_level,
        "bond_requirement": bond_requirement,
        "layoff_history": layoff_history,
        "relocation_willingness": relocation_willingness,
        "previous_ctc_lpa": previous_ctc_lpa,
        "expected_ctc_lpa": expected_ctc_lpa,
        "notice_period_days": notice_period_days,
        "employment_gap_months": employment_gap_months
    }])
    candidate = candidate[X_columns] if False else candidate
    prediction = model.predict(candidate)[0]
    st.subheader("4. Prediction Result")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(candidate)[0]
        classes = list(model.named_steps["classifier"].classes_)
        placed_index = classes.index("placed")
        probability = probabilities[placed_index]
        st.metric(
            "Estimated Placement Probability",
            f"{probability:.1%}"
        )
        st.progress(float(probability))
    if prediction == "placed":
        st.success("Prediction: Placed")
    else:
        st.warning("Prediction: Not Placed")
    st.caption(
        "This is a model estimate, not a guarantee. "
        "Use it as a decision-support tool, not as the sole "
        "basis for hiring decisions."
    )
