import streamlit as st

from pathlib import Path
import sys


# --------------------------------------------------
# Project setup
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predictor import analyze_message


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Phishing & Scam Detector",
    page_icon="🛡️",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛡️ Phishing & Scam Detector")

st.write(
    "Analyze a suspicious message using machine learning "
    "and security-based phishing indicators."
)


# --------------------------------------------------
# Message input
# --------------------------------------------------

message = st.text_area(
    "Paste your message below",
    placeholder=(
        "Paste a suspicious SMS, email, or message here..."
    ),
    height=180
)


# --------------------------------------------------
# Analyze
# --------------------------------------------------

if st.button(
    "🔍 Analyze Message",
    use_container_width=True
):

    if not message.strip():

        st.warning(
            "Please enter a message first."
        )

    else:
        analysis = analyze_message(message)

    prediction = analysis["prediction"]
    ml_risk = analysis["ml_risk_percentage"]
    security_score = analysis["security_score"]
    overall = analysis["overall_assessment"]

    # Your existing result display
    # ...

    # URL Intelligence
    st.subheader("🔗 URL Intelligence")

    if analysis["url_findings"]:
        for finding in analysis["url_findings"]:
            st.warning(finding)
    else:
        st.success("No suspicious URL characteristics detected.")

    # URL Phishing Model
    st.subheader("🧠 URL Phishing Model")

    if analysis["url_predictions"]:
        for url_result in analysis["url_predictions"]:
            st.write("**URL:**", url_result["url"])

            if url_result["prediction"] == "PHISHING":
                st.error(
                    f"🚨 PHISHING — "
                    f"{url_result['phishing_probability']}% phishing probability"
                )
            else:
                st.success(
                    f"✅ LEGITIMATE — "
                    f"{url_result['phishing_probability']}% phishing probability"
                )
    else:
        st.info("No URL detected in this message.")


        # --------------------------------------------------
        # Recommendation
        # --------------------------------------------------

    st.subheader("🛡️ Safety Recommendation")

    if overall == "HIGH RISK":

            st.error(
                "Avoid clicking links or sharing OTPs, "
                "passwords, banking details, or other "
                "sensitive information. Verify the message "
                "through an official channel."
            )

    elif overall == "SUSPICIOUS":

            st.warning(
                "Treat this message with caution. Do not "
                "share sensitive information until you "
                "independently verify the sender or request."
            )

    else:

            st.info(
                "No major phishing indicators were detected. "
                "Still verify unexpected requests independently."
            )


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.divider()

st.caption(
    "This tool provides an automated risk assessment. "
    "It should not be treated as definitive proof that "
    "a message is malicious or legitimate."
)