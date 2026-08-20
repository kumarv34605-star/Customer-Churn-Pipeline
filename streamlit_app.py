import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Theme (appearance only — no logic affected)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ---- Hide Streamlit default chrome ---- */
        #MainMenu, footer, header {visibility: hidden !important;}
        .stDeployButton {display: none !important;}

        /* ---- Base ---- */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI",
                Roboto, Helvetica, Arial, sans-serif;
        }
        .stApp {
            background:
                radial-gradient(1200px 500px at 10% -10%, #e0ecff 0%, rgba(224,236,255,0) 60%),
                radial-gradient(1000px 500px at 95% 0%, #e8e3ff 0%, rgba(232,227,255,0) 55%),
                #f5f7fb;
        }
        .block-container {
            padding-top: 2.2rem !important;
            padding-bottom: 3rem !important;
            max-width: 1250px;
        }

        /* ---- Hero banner ---- */
        .hero {
            background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 45%, #6d28d9 100%);
            border-radius: 20px;
            padding: 1.9rem 2.1rem;
            color: #ffffff;
            box-shadow: 0 18px 40px -22px rgba(37, 99, 235, 0.85);
            margin-bottom: 1.6rem;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(255,255,255,0.16);
            border: 1px solid rgba(255,255,255,0.28);
            color: #eef2ff;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.15;
            margin: 0 0 0.45rem 0;
        }
        .hero-sub {
            font-size: 1rem;
            color: #dbeafe;
            max-width: 720px;
            margin: 0;
            line-height: 1.55;
        }

        /* ---- Section headers ---- */
        .section-header {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 1.08rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0.1rem 0 0.35rem 0;
            padding-left: 0.8rem;
            border-left: 4px solid #2563eb;
        }
        .section-sub {
            font-size: 0.84rem;
            color: #64748b;
            margin: 0 0 1rem 0.8rem;
        }

        /* ---- Cards ---- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #e6ebf2 !important;
            border-radius: 16px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04),
                        0 14px 34px -22px rgba(15, 23, 42, 0.35);
            padding: 1.45rem 1.7rem !important;
            margin-bottom: 1.15rem !important;
        }

        /* ---- Inputs ---- */
        .stSelectbox label, .stNumberInput label {
            font-weight: 600 !important;
            color: #334155 !important;
            font-size: 0.86rem !important;
            margin-bottom: 0.25rem !important;
        }
        div[data-baseweb="select"] > div,
        .stNumberInput input {
            border-radius: 10px !important;
            border-color: #dbe3ec !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        /* Selectbox text */
        div[data-baseweb="select"] span {
            color: #0f172a !important;
        }

        /* Number input text */
        .stNumberInput input {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        /* Placeholder text */
        .stNumberInput input::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        /* Hover */
        div[data-baseweb="select"] > div:hover,
        .stNumberInput input:hover {
            border-color: #93b4fb !important;
        }

        /* ---- Submit button ---- */
        .stFormSubmitButton > button {
            background: linear-gradient(90deg, #2563eb, #6d28d9) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            font-size: 1.02rem !important;
            letter-spacing: 0.01em;
            padding: 0.85rem 1rem !important;
            box-shadow: 0 12px 26px -12px rgba(76, 61, 220, 0.85) !important;
            transition: transform 0.16s ease, box-shadow 0.16s ease !important;
        }
        .stFormSubmitButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 32px -12px rgba(76, 61, 220, 0.95) !important;
        }
        .stFormSubmitButton > button:active { transform: translateY(0); }

        /* ---- Result cards ---- */
        .result {
            border-radius: 16px;
            padding: 1.5rem 1.7rem;
            margin-top: 0.4rem;
            display: flex;
            gap: 1rem;
            align-items: flex-start;
        }
        .result-icon { font-size: 1.9rem; line-height: 1; }
        .result-title { font-size: 1.22rem; font-weight: 750; margin: 0 0 0.3rem 0; }
        .result-text { font-size: 0.93rem; margin: 0; line-height: 1.55; }
        .result-risk {
            background: linear-gradient(135deg, #fff1f2, #ffe4e6);
            border: 1px solid #fecdd3;
            color: #881337;
        }
        .result-safe {
            background: linear-gradient(135deg, #ecfdf5, #d1fae5);
            border: 1px solid #a7f3d0;
            color: #064e3b;
        }
        .result-fail {
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #7c2d12;
        }
        .stAlert { border-radius: 12px !important; }

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #111c33 100%);
            border-right: 1px solid #1e293b;
        }
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label {
            color: #e2e8f0 !important;
        }
        .sb-brand {
            font-size: 1.22rem; font-weight: 800; color: #ffffff;
            letter-spacing: -0.01em; margin-bottom: 0.15rem;
        }
        .sb-tag { font-size: 0.82rem; color: #94a3b8; margin-top: 0; }
        .sb-note { font-size: 0.83rem; color: #cbd5e1; line-height: 1.6; }
        .sb-step {
            display: flex; gap: 0.6rem; align-items: flex-start;
            font-size: 0.83rem; color: #cbd5e1; margin-bottom: 0.55rem;
        }
        .sb-num {
            flex: 0 0 auto; width: 22px; height: 22px; border-radius: 999px;
            background: rgba(37,99,235,0.22); border: 1px solid #3b82f6;
            color: #bfdbfe; font-size: 0.72rem; font-weight: 700;
            display: flex; align-items: center; justify-content: center;
        }
        .sb-divider { height: 1px; background: #1e293b; margin: 1.1rem 0; }

        /* ---- Footer note ---- */
        .foot {
            text-align: center; font-size: 0.78rem; color: #94a3b8;
            margin-top: 1.6rem;
        }
        /* Prediction details */
        [data-testid="stAlert"] + div p {
            color: #0f172a !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar — branded intro
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<p class="sb-brand">📊 Churn Predictor</p>'
        '<p class="sb-tag">Customer retention intelligence</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sb-note">'
        "Fill in the customer profile and run a prediction to estimate churn "
        "risk. Every field maps directly to your trained model."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sb-note" style="font-weight:700;color:#ffffff;margin-bottom:0.7rem;">'
        "How it works</p>"
        '<div class="sb-step"><div class="sb-num">1</div>'
        "<div>Enter customer, service &amp; account details.</div></div>"
        '<div class="sb-step"><div class="sb-num">2</div>'
        "<div>Click <b>Predict Churn</b>.</div></div>"
        '<div class="sb-step"><div class="sb-num">3</div>'
        "<div>Review the risk outcome and act on it.</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sb-note" style="color:#64748b;">'
        "Predictions are model estimates and should support — not replace — "
        "human judgement.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Hero
# ---------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<span class="hero-badge">Machine learning · Retention</span>'
    '<div class="hero-title">Customer Churn Prediction</div>'
    '<p class="hero-sub">Enter customer information below to predict whether '
    "the customer is likely to churn, and spot at-risk accounts before they "
    "leave.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Prediction Form
# ---------------------------------------------------------
with st.form("customer_form"):
    # =====================================================
    # Customer Information
    # =====================================================
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">👤 Customer Information</div>'
            '<p class="section-sub">Basic demographic profile of the customer.</p>',
            unsafe_allow_html=True,
        )
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            gender = st.selectbox(
                "Gender", ["Male", "Female"], help="Customer's recorded gender."
            )

        with col2:
            SeniorCitizen = st.selectbox(
                "Senior Citizen", [0, 1], help="0 = No, 1 = Yes (65 or older)."
            )

        with col3:
            Partner = st.selectbox(
                "Partner", ["Yes", "No"], help="Does the customer have a partner?"
            )

        with col4:
            Dependents = st.selectbox(
                "Dependents", ["Yes", "No"], help="Does the customer have dependents?"
            )

    # =====================================================
    # Service Information
    # =====================================================
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">📱 Service Information</div>'
            '<p class="section-sub">Products and add-ons the customer subscribes to.</p>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
            MultipleLines = st.selectbox(
                "Multiple Lines", ["Yes", "No", "No phone service"]
            )
            InternetService = st.selectbox(
                "Internet Service", ["DSL", "Fiber optic", "No"]
            )
            OnlineSecurity = st.selectbox(
                "Online Security", ["Yes", "No", "No internet service"]
            )
            OnlineBackup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )

        with col2:
            DeviceProtection = st.selectbox(
                "Device Protection", ["Yes", "No", "No internet service"]
            )
            TechSupport = st.selectbox(
                "Tech Support", ["Yes", "No", "No internet service"]
            )
            StreamingTV = st.selectbox(
                "Streaming TV", ["Yes", "No", "No internet service"]
            )
            StreamingMovies = st.selectbox(
                "Streaming Movies", ["Yes", "No", "No internet service"]
            )

        with col3:
            Contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                help="Month-to-month contracts typically carry higher churn risk.",
            )
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
            PaymentMethod = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

    # =====================================================
    # Account Information
    # =====================================================
    with st.container(border=True):
        st.markdown(
            '<div class="section-header">💰 Account Information</div>'
            '<p class="section-sub">Tenure and billing figures for this account.</p>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=24,
                step=1,
                help="How many months the customer has stayed with the company.",
            )

        with col2:
            MonthlyCharges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=60.0,
                step=1.0,
                help="Amount charged to the customer each month.",
            )

        with col3:
            TotalCharges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=1400.0,
                step=1.0,
                help="Total amount charged over the customer's lifetime.",
            )

    # =====================================================
    # Prediction Button
    # =====================================================
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    try:
        customer_data = CustomData(
            gender=gender,
            SeniorCitizen=SeniorCitizen,
            Partner=Partner,
            Dependents=Dependents,
            tenure=tenure,
            PhoneService=PhoneService,
            MultipleLines=MultipleLines,
            InternetService=InternetService,
            OnlineSecurity=OnlineSecurity,
            OnlineBackup=OnlineBackup,
            DeviceProtection=DeviceProtection,
            TechSupport=TechSupport,
            StreamingTV=StreamingTV,
            StreamingMovies=StreamingMovies,
            Contract=Contract,
            PaperlessBilling=PaperlessBilling,
            PaymentMethod=PaymentMethod,
            MonthlyCharges=MonthlyCharges,
            TotalCharges=TotalCharges,
        )

        customer_df = customer_data.get_data_as_data_frame()

        with st.spinner("Analyzing customer data…"):
            prediction_pipeline = PredictPipeline()

            prediction, probability = (
                prediction_pipeline.predict_with_probability(
                    customer_df
                )
            )
        risk_percentage = probability * 100
        
        if risk_percentage < 30:
            risk_level = "LOW RISK"
        elif risk_percentage < 60:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "HIGH RISK"

        if prediction[0] == 1:

            st.html(
                f"""
                <div style="
                    background: #fff1f2;
                    border: 1px solid #fecdd3;
                    border-radius: 16px;
                    padding: 24px;
                    margin-top: 20px;
                ">

                    <div style="
                        font-size: 22px;
                        font-weight: 700;
                        color: #991b1b;
                        margin-bottom: 18px;
                    ">
                        ⚠️ Customer is likely to churn
                    </div>

                    <div style="
                        color: #0f172a;
                        font-size: 16px;
                        font-weight: 600;
                        margin-bottom: 8px;
                    ">
                        Churn Risk
                    </div>

                    <div style="
                        color: #0f172a;
                        font-size: 32px;
                        font-weight: 750;
                        margin-bottom: 10px;
                    ">
                        {risk_percentage:.2f}%
                    </div>

                    <div style="
                        height: 10px;
                        background: #fee2e2;
                        border-radius: 10px;
                        overflow: hidden;
                        margin-bottom: 12px;
                    ">
                        <div style="
                            width: {min(risk_percentage, 100)}%;
                            height: 100%;
                            background: #dc2626;
                            border-radius: 10px;
                        "></div>
                    </div>

                    <div style="
                        color: #dc2626;
                        font-size: 16px;
                        font-weight: 700;
                        margin-bottom: 12px;
                    ">
                        {risk_level}
                    </div>

                    <div style="
                        color: #475569;
                        font-size: 14px;
                        line-height: 1.6;
                    ">
                        This profile matches patterns seen in customers
                        who left. Consider a retention offer, contract
                        upgrade, or proactive support check-in.
                    </div>

                </div>
                """
            )

        else:

            st.html(
                f"""
                <div style="
                    background: #f0fdf4;
                    border: 1px solid #bbf7d0;
                    border-radius: 16px;
                    padding: 24px;
                    margin-top: 20px;
                ">

                    <div style="
                        font-size: 22px;
                        font-weight: 700;
                        color: #166534;
                        margin-bottom: 18px;
                    ">
                        ✅ Customer is unlikely to churn
                    </div>

                    <div style="
                        color: #0f172a;
                        font-size: 16px;
                        font-weight: 600;
                        margin-bottom: 8px;
                    ">
                        Churn Risk
                    </div>

                    <div style="
                        color: #0f172a;
                        font-size: 32px;
                        font-weight: 750;
                        margin-bottom: 10px;
                    ">
                        {risk_percentage:.2f}%
                    </div>

                    <div style="
                        height: 10px;
                        background: #dcfce7;
                        border-radius: 10px;
                        overflow: hidden;
                        margin-bottom: 12px;
                    ">
                        <div style="
                            width: {min(risk_percentage, 100)}%;
                            height: 100%;
                            background: #16a34a;
                            border-radius: 10px;
                        "></div>
                    </div>

                    <div style="
                        color: #16a34a;
                        font-size: 16px;
                        font-weight: 700;
                        margin-bottom: 12px;
                    ">
                        {risk_level}
                    </div>

                    <div style="
                        color: #475569;
                        font-size: 14px;
                        line-height: 1.6;
                    ">
                        This profile looks stable. Keep the experience
                        consistent and consider upsell opportunities.
                    </div>

                </div>
                """
            )

    except Exception as e:
        st.markdown(
            '<div class="result result-fail">'
            '<div class="result-icon">🚫</div>'
            "<div><p class='result-title'>Prediction failed</p>"
            f"<p class='result-text'>{str(e)}</p></div>"
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown(
    '<p class="foot">Customer Churn Prediction · powered by your trained '
    "machine learning pipeline</p>",
    unsafe_allow_html=True,
)