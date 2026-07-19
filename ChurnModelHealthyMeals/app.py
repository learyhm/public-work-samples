import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ============================================================
# Load model + encoder once
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("churn_xgb_healthy_meals.pkl", "rb") as f:
        model = pickle.load(f)
    with open("churn_encoder_healthy_meals.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, encoder

model, encoder = load_artifacts()

# ============================================================
# Helper: dynamic subscription + renewal checkbox
# ============================================================
def subscription_with_renewal(label_sub, label_renew, key_sub, key_renew):
    """
    Creates a subscription checkbox and a dynamically disabled renewal checkbox.
    Automatically resets renewal to False if subscription is unchecked.
    """
    sub_value = st.checkbox(label_sub, key=key_sub)

    # If subscription is unchecked → force renewal False
    if not sub_value:
        st.session_state[key_renew] = False

    renew_value = st.checkbox(
        label_renew,
        key=key_renew,
        disabled=not sub_value
    )

    return sub_value, renew_value


# ============================================================
# UI
# ============================================================

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

# -------------------------
# Demographics
# -------------------------
st.header("🧍 Demographics")

age = st.slider("Age", min_value=18, max_value=100, value=35)
tech_comfort_score = st.slider("Tech Comfort Score", min_value=1, max_value=10, value=5)

income_level = st.radio(
    "Income Level",
    ["Low", "Medium", "High", "Very High"],
    index=1
)

education = st.radio(
    "Education",
    ["Graduate", "High School", "Other", "Post-Graduate"],
    index=0
)

device_type = st.radio(
    "Device Type",
    ["Desktop-only", "Mobile-only", "Multi-device"],
    index=2
)

# -------------------------
# Engagement Features
# -------------------------
st.header("📈 Engagement Features")

num_session = st.slider("Total Sessions (Past Year)", 0, 300, 27)
gross_sessions_length = st.slider("Gross Session Length (Minutes)", 0, 20000, 1500)
num_active_days = st.slider("Active Days", 0, 365, 45)
num_active_quarters = st.slider("Active Quarters in 2022", 0, 4, 2)
avg_sessions_per_quarter = st.slider("Avg Sessions per Quarter", 0, 100, 10)
active_q4 = st.checkbox("Active in Q4", value=True)
avg_session_length = st.slider("Avg Session Length (Minutes)", 0, 300, 20)
engagement_trend = st.slider("Engagement Trend", -1.0, 1.0, 0.1)
session_frequency = st.slider("Session Frequency", 0, 50, 5)

# -------------------------
# Subscription Flags (Dynamic disabling)
# -------------------------
st.header("📦 Subscription Flags")
st.markdown("**Renewed options are automatically disabled unless Subscription is selected.**")

mindful_living_subscription, mindful_living_renew = subscription_with_renewal(
    "Mindful Living Subscription",
    "Mindful Living Renewed",
    key_sub="ml_sub",
    key_renew="ml_renew"
)

premium_health_subscription, premium_health_renew = subscription_with_renewal(
    "Premium Health Subscription",
    "Premium Health Renewed",
    key_sub="ph_sub",
    key_renew="ph_renew"
)

wellness_tracker_subscription, wellness_tracker_renew = subscription_with_renewal(
    "Wellness Tracker Subscription",
    "Wellness Tracker Renewed",
    key_sub="wt_sub",
    key_renew="wt_renew"
)

daily_fitness_subscription, daily_fitness_renew = subscription_with_renewal(
    "Daily Fitness Subscription",
    "Daily Fitness Renewed",
    key_sub="df_sub",
    key_renew="df_renew"
)

# -------------------------
# Renewal History
# -------------------------
st.header("🔄 Renewal History")

prior_renewals = st.slider("Prior Renewals", 0, 10, 1)

# ============================================================
# Predict Button
# ============================================================
if st.button("Predict"):

    # -----------------------------------------------------------
    # Build categorical DataFrame — must match encoder exactly
    # -----------------------------------------------------------
    raw = pd.DataFrame([{
        "INCOME_LEVEL": income_level,
        "EDUCATION": education,
        "DEVICE_TYPE": device_type
    }])

    encoded = encoder.transform(raw)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    # -----------------------------------------------------------
    # Numeric + flag features
    # -----------------------------------------------------------
    numeric_df = pd.DataFrame([{
        "NUM_SESSION": num_session,
        "GROSS_SESSIONS_LENGTH": gross_sessions_length,
        "NUM_ACTIVE_DAYS": num_active_days,
        "NUMACTIVEQUARTERS2022": num_active_quarters,
        "AVGSESSIONSPERQUARTER": avg_sessions_per_quarter,
        "ACTIVEQ4": int(active_q4),
        "AVG_SESSION_LENGTH": avg_session_length,
        "ENGAGEMENT_TREND": engagement_trend,
        "SESSION_FREQUENCY": session_frequency,
        "MINDFULLIVINGSUBSCRIPTION": int(mindful_living_subscription),
        "MINDFULLIVINGRENEW": int(mindful_living_renew),
        "PREMIUMHEALTHSUBSCRIPTION": int(premium_health_subscription),
        "PREMIUMHEALTHRENEW": int(premium_health_renew),
        "WELLNESSTRACKERSUBSCRIPTION": int(wellness_tracker_subscription),
        "WELLNESSTRACKERRENEW": int(wellness_tracker_renew),
        "DAILYFITNESSSUBSCRIPTION": int(daily_fitness_subscription),
        "DAILYFITNESSRENEW": int(daily_fitness_renew),
        "PRIORRENEWALS": prior_renewals,
        "AGE": age,
        "TECH_COMFORT_SCORE": tech_comfort_score
    }])

    # -----------------------------------------------------------
    # Combine numeric + encoded categorical features
    # -----------------------------------------------------------
    input_df = pd.concat([numeric_df, encoded_df], axis=1)

    # -----------------------------------------------------------
    # Predict
    # -----------------------------------------------------------
    probability = model.predict_proba(input_df)[0][1]

    risk = (
        "Low" if probability >= 0.6
        else "Medium" if probability >= 0.4
        else "High"
    )

    # -----------------------------------------------------------
    # Display results
    # -----------------------------------------------------------
    st.metric("Renewal Probability", f"{probability:.2f}")

    if risk == "High":
        st.error(f"Churn Risk: {risk}")
    elif risk == "Medium":
        st.warning(f"Churn Risk: {risk}")
    else:
        st.success(f"Churn Risk: {risk}")
