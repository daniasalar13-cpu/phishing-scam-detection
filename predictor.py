import re
import joblib
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# Existing SMS model
model = joblib.load(MODELS_DIR / "phishing_scam_model.pkl")
vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
threshold = joblib.load(MODELS_DIR / "threshold.pkl")


# New URL phishing model
URL_MODEL_PATH = MODELS_DIR / "url_phishing_model.pkl"
URL_FEATURES_PATH = MODELS_DIR / "url_feature_columns.pkl"
URL_THRESHOLD_PATH = MODELS_DIR / "url_threshold.pkl"

url_phishing_model = joblib.load(URL_MODEL_PATH)
url_feature_columns = joblib.load(URL_FEATURES_PATH)
url_phishing_threshold = joblib.load(URL_THRESHOLD_PATH)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# --------------------------------------------------
# Load trained ML components
# --------------------------------------------------

model = joblib.load(
    MODELS_DIR / "phishing_scam_model.pkl"
)

vectorizer = joblib.load(
    MODELS_DIR / "tfidf_vectorizer.pkl"
)

threshold = joblib.load(
    MODELS_DIR / "threshold.pkl"
)


# --------------------------------------------------
# Feature engineering
# --------------------------------------------------

def extract_features(messages):
    features = pd.DataFrame()

    features["message_length"] = messages.apply(len)

    features["word_count"] = messages.apply(
        lambda x: len(x.split())
    )

    features["digit_count"] = messages.apply(
        lambda x: sum(c.isdigit() for c in x)
    )

    features["exclamation_count"] = messages.apply(
        lambda x: x.count("!")
    )

    features["question_count"] = messages.apply(
        lambda x: x.count("?")
    )

    features["currency_count"] = messages.apply(
        lambda x: len(re.findall(r"[₹$€£]", x))
    )

    features["url_count"] = messages.apply(
        lambda x: len(
            re.findall(
                r"(https?://\S+|www\.\S+)",
                x,
                re.IGNORECASE
            )
        )
    )

    features["phone_number_count"] = messages.apply(
        lambda x: len(
            re.findall(
                r"\b\d{7,15}\b",
                x
            )
        )
    )

    return features


# --------------------------------------------------
# ML prediction
# --------------------------------------------------

def predict_message(message):

    message_series = pd.Series([message])

    # TF-IDF features
    tfidf_features = vectorizer.transform(
        message_series
    )

    # Engineered features
    engineered_features = extract_features(
        message_series
    )

    engineered_features = engineered_features.values

    # Combine both feature sets
    combined_features = hstack([
        tfidf_features,
        engineered_features
    ])

    # Scam probability
    probability = model.predict_proba(
        combined_features
    )[0][1]

    # Apply final threshold
    prediction = 1 if probability >= threshold else 0

    if prediction == 1:
        result = "SCAM / SPAM"
    else:
        result = "SAFE"

    return result, probability


# --------------------------------------------------
# Warning-sign detection
# --------------------------------------------------

def detect_warning_signs(message):

    warnings = []

    text = message.lower()

    # --------------------------------------------------
    # URL detection
    # --------------------------------------------------

    if re.search(
        r"(https?://\S+|www\.\S+)",
        message,
        re.IGNORECASE
    ):
        warnings.append(
            "Contains a URL"
        )


    # --------------------------------------------------
    # Currency / financial information
    # --------------------------------------------------

    if re.search(r"[₹$€£]", message):

        warnings.append(
            "Contains financial/currency information"
        )


    # --------------------------------------------------
    # Phone number
    # --------------------------------------------------

    if re.search(
        r"\b\d{7,15}\b",
        message
    ):

        warnings.append(
            "Contains a phone number"
        )


    # --------------------------------------------------
    # Urgency / pressure
    # --------------------------------------------------

    urgency_words = [
        "urgent",
        "immediately",
        "act now",
        "hurry",
        "expires",
        "expired",
        "limited time",
        "asap",
        "within 24 hours",
        "today only",
        "last chance"
    ]

    if any(word in text for word in urgency_words):

        warnings.append(
            "Uses urgency or pressure language"
        )


    # --------------------------------------------------
    # Prize / reward scam
    # --------------------------------------------------

    reward_words = [
        "won",
        "winner",
        "prize",
        "reward",
        "lottery",
        "congratulations",
        "cash prize",
        "free gift",
        "lucky winner",
        "claim your prize"
    ]

    if any(word in text for word in reward_words):

        warnings.append(
            "Contains prize/reward language"
        )


    # --------------------------------------------------
    # Account / login phishing
    # --------------------------------------------------

    account_words = [
        "verify your account",
        "verify account",
        "account verification",
        "account suspended",
        "account blocked",
        "account will be closed",
        "confirm your account",
        "login",
        "log in",
        "sign in"
    ]

    if any(word in text for word in account_words):

        warnings.append(
            "Contains account or login-related language"
        )

    # KYC / identity verification
    kyc_words = [
    "kyc",
    "know your customer",
    "identity verification",
    "verification expired",
    "verification has expired"
    ]

    if any(word in text for word in kyc_words):
        warnings.append(
        "Contains KYC or identity-verification language"
    )


    # --------------------------------------------------
    # Sensitive information requests
    # --------------------------------------------------

    sensitive_words = [
        "password",
        "otp",
        "one time password",
        "verification code",
        "security code",
        "pin",
        "cvv",
        "credit card",
        "debit card",
        "bank account"
    ]

    if any(word in text for word in sensitive_words):

        warnings.append(
            "Mentions or requests sensitive information"
        )


    # --------------------------------------------------
    # Banking / payment language
    # --------------------------------------------------

    payment_words = [
        "bank transfer",
        "send money",
        "transfer money",
        "make a payment",
        "payment required",
        "pay now",
        "pay immediately",
        "upi",
        "refund",
        "transaction"
    ]

    if any(word in text for word in payment_words):

        warnings.append(
            "Contains banking or payment-related language"
        )


    # --------------------------------------------------
    # Impersonation indicators
    # --------------------------------------------------

    impersonation_words = [
        "bank support",
        "customer support",
        "security team",
        "technical support",
        "government",
        "income tax department",
        "police department",
        "official notice",
        "your bank"
    ]

    if any(word in text for word in impersonation_words):

        warnings.append(
            "May be impersonating an organization or authority"
        )


    return warnings

def extract_url_features(url):
    """
    Extract the same 18 URL features used during model training.
    This does not visit or fetch the URL.
    """

    url = str(url).strip()

    parse_url = url

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        parse_url
    ):
        parse_url = "//" + parse_url

    parsed = urlparse(parse_url)

    hostname = parsed.hostname or ""

    # Count letters and digits
    letters = sum(char.isalpha() for char in url)
    digits = sum(char.isdigit() for char in url)

    # Count special characters
    special_characters = sum(
        not char.isalnum()
        for char in url
    )

    # Detect IPv4 address
    is_ip = int(
        bool(
            re.fullmatch(
                r"\d{1,3}(\.\d{1,3}){3}",
                hostname
            )
        )
    )

    # Domain information
    domain_parts = hostname.split(".") if hostname else []

    if len(domain_parts) >= 2:
        subdomain_count = max(0, len(domain_parts) - 2)
        tld_length = len(domain_parts[-1])
    else:
        subdomain_count = 0
        tld_length = 0

    # Detect percent-encoded characters
    percent_encoded_count = len(
        re.findall(
            r"%[0-9A-Fa-f]{2}",
            url
        )
    )

    # Detect URL obfuscation
    has_obfuscation = int(
        percent_encoded_count > 0
        or "@" in url
        or "\\" in url
    )

    # Ratios
    url_length = len(url)

    letter_ratio = (
        letters / url_length
        if url_length > 0 else 0
    )

    digit_ratio = (
        digits / url_length
        if url_length > 0 else 0
    )

    special_ratio = (
        special_characters / url_length
        if url_length > 0 else 0
    )

    return {
        "url_length": url_length,
        "domain_length": len(hostname),
        "is_ip": is_ip,
        "tld_length": tld_length,
        "subdomain_count": subdomain_count,
        "is_https": int(
            parsed.scheme.lower() == "https"
        ),
        "letter_count": letters,
        "letter_ratio": letter_ratio,
        "digit_count": digits,
        "digit_ratio": digit_ratio,
        "equals_count": url.count("="),
        "question_count": url.count("?"),
        "ampersand_count": url.count("&"),
        "at_count": url.count("@"),
        "hyphen_count": url.count("-"),
        "dot_count": url.count("."),
        "percent_encoded_count": percent_encoded_count,
        "special_char_ratio": special_ratio
    }

def extract_url_features(url):
    """
    Extract the same 18 URL features used during model training.
    This does not visit or fetch the URL.
    """

    url = str(url).strip()

    parse_url = url

    if not re.match(
        r"^[a-zA-Z][a-zA-Z0-9+.-]*://",
        parse_url
    ):
        parse_url = "//" + parse_url

    parsed = urlparse(parse_url)

    hostname = parsed.hostname or ""

    # Count letters and digits
    letters = sum(char.isalpha() for char in url)
    digits = sum(char.isdigit() for char in url)

    # Count special characters
    special_characters = sum(
        not char.isalnum()
        for char in url
    )

    # Detect IPv4 address
    is_ip = int(
        bool(
            re.fullmatch(
                r"\d{1,3}(\.\d{1,3}){3}",
                hostname
            )
        )
    )

    # Domain information
    domain_parts = hostname.split(".") if hostname else []

    if len(domain_parts) >= 2:
        subdomain_count = max(0, len(domain_parts) - 2)
        tld_length = len(domain_parts[-1])
    else:
        subdomain_count = 0
        tld_length = 0

    # Detect percent-encoded characters
    percent_encoded_count = len(
        re.findall(
            r"%[0-9A-Fa-f]{2}",
            url
        )
    )

    # Detect URL obfuscation
    has_obfuscation = int(
        percent_encoded_count > 0
        or "@" in url
        or "\\" in url
    )

    # Ratios
    url_length = len(url)

    letter_ratio = (
        letters / url_length
        if url_length > 0 else 0
    )

    digit_ratio = (
        digits / url_length
        if url_length > 0 else 0
    )

    special_ratio = (
        special_characters / url_length
        if url_length > 0 else 0
    )

    return {
        "url_length": url_length,
        "domain_length": len(hostname),
        "is_ip": is_ip,
        "tld_length": tld_length,
        "subdomain_count": subdomain_count,
        "is_https": int(
            parsed.scheme.lower() == "https"
        ),
        "letter_count": letters,
        "letter_ratio": letter_ratio,
        "digit_count": digits,
        "digit_ratio": digit_ratio,
        "equals_count": url.count("="),
        "question_count": url.count("?"),
        "ampersand_count": url.count("&"),
        "at_count": url.count("@"),
        "hyphen_count": url.count("-"),
        "dot_count": url.count("."),
        "percent_encoded_count": percent_encoded_count,
        "special_char_ratio": special_ratio
    }

def predict_url_phishing(url):
    """
    Predict whether a URL is phishing using the custom URL model.
    """

    features = extract_url_features(url)

    feature_data = pd.DataFrame([features])

    feature_data = feature_data[url_feature_columns]

    phishing_probability = (
        url_phishing_model
        .predict_proba(feature_data)[0][0]
    )

    is_phishing = (
        phishing_probability >= url_phishing_threshold
    )

    return {
        "is_phishing": bool(is_phishing),
        "phishing_probability": round(
            phishing_probability * 100,
            2
        )
    }

def analyze_url_predictions(message):
    """
    Find URLs in a message and run the phishing model on each URL.
    """

    url_pattern = r"(https?://\S+|www\.\S+)"

    urls = re.findall(
        url_pattern,
        message,
        re.IGNORECASE
    )

    predictions = []

    for url in urls:
        result = predict_url_phishing(url)

        predictions.append({
            "url": url,
            "phishing_probability": result["phishing_probability"],
            "prediction": (
                "PHISHING"
                if result["is_phishing"]
                else "LEGITIMATE"
            )
        })

    return predictions

def analyze_urls(message):
    """
    Analyze URLs found in a message without visiting them.
    Returns a list of URL-related security signals.
    """

    url_pattern = r"(https?://\S+|www\.\S+)"

    urls = re.findall(
        url_pattern,
        message,
        re.IGNORECASE
    )

    findings = []

    for url in urls:

        # Add a scheme so urlparse can properly identify the host
        parsed_url = urlparse(
            url if url.startswith(("http://", "https://"))
            else "http://" + url
        )

        hostname = parsed_url.hostname

        if not hostname:
            findings.append(
                "URL could not be parsed safely"
            )
            continue

        hostname = hostname.lower()

        # ----------------------------------------------
        # IP address instead of domain
        # ----------------------------------------------

        if re.fullmatch(
            r"\d{1,3}(\.\d{1,3}){3}",
            hostname
        ):
            findings.append(
                "URL uses an IP address instead of a domain name"
            )


        # ----------------------------------------------
        # Very long URL
        # ----------------------------------------------

        if len(url) > 100:
            findings.append(
                "URL is unusually long"
            )


        # ----------------------------------------------
        # User information inside URL
        # ----------------------------------------------

        if "@" in url:
            findings.append(
                "URL contains an @ symbol"
            )


        # ----------------------------------------------
        # Many subdomains
        # ----------------------------------------------

        parts = hostname.split(".")

        if len(parts) >= 4:
            findings.append(
                "URL contains an unusually large number of subdomains"
            )


        # ----------------------------------------------
        # Suspicious URL keywords
        # ----------------------------------------------

        suspicious_words = [
            "login",
            "verify",
            "verification",
            "account",
            "secure",
            "security",
            "update",
            "password",
            "wallet",
            "payment",
            "confirm"
        ]

        if any(
            word in url.lower()
            for word in suspicious_words
        ):
            findings.append(
                "URL contains security-sensitive keywords"
            )


    return findings

def calculate_security_score(message):
    score = 0

    message_lower = message.lower()

    # Existing warning-sign scoring
    if re.search(r"(https?://\S+|www\.\S+)", message, re.IGNORECASE):
        score += 25

    urgency_words = [
        "urgent", "immediately", "act now", "hurry", "expires",
        "expired", "limited time", "asap", "within 24 hours",
        "today only", "last chance"
    ]
    if any(word in message_lower for word in urgency_words):
        score += 15

    account_words = [
        "verify your account", "verify account", "account verification",
        "account suspended", "account will be closed",
        "account blocked", "account will be blocked",
        "account will be suspended", "account has been suspended",
        "confirm your account", "login", "log in", "sign in"
    ]
    if any(word in message_lower for word in account_words):
        score += 20

    sensitive_words = [
        "password", "otp", "one time password", "verification code",
        "security code", "pin", "cvv", "credit card",
        "debit card", "bank account"
    ]
    if any(word in message_lower for word in sensitive_words):
        score += 25

    payment_words = [
        "bank transfer", "send money", "transfer money",
        "make a payment", "payment required", "pay now",
        "pay immediately", "upi", "refund", "transaction"
    ]
    if any(word in message_lower for word in payment_words):
        score += 20

    reward_words = [
        "won", "winner", "prize", "reward", "lottery",
        "congratulations", "cash prize", "free gift",
        "lucky winner", "claim your prize"
    ]
    if any(word in message_lower for word in reward_words):
        score += 20

    impersonation_words = [
        "bank support", "customer support", "security team",
        "technical support", "government", "income tax department",
        "police department", "official notice", "your bank"
    ]
    if any(word in message_lower for word in impersonation_words):
        score += 15

    currency_pattern = r"[$₹€£]|rs\.?|inr"
    if re.search(currency_pattern, message_lower):
        score += 10

    kyc_words = [
        "kyc", "know your customer", "identity verification",
        "verification expired", "verification has expired"
    ]
    if any(word in message_lower for word in kyc_words):
        score += 20

    # URL intelligence scoring
    url_findings = analyze_urls(message)

    for finding in url_findings:
        if "IP address" in finding:
            score += 25

        elif "unusually long" in finding:
            score += 10

        elif "@ symbol" in finding:
            score += 20

        elif "large number of subdomains" in finding:
            score += 15

        elif "security-sensitive keywords" in finding:
            score += 15

        elif "could not be parsed" in finding:
            score += 20

    # Keep score between 0 and 100
    return min(score, 100)

# --------------------------------------------------
# Complete message analysis
# --------------------------------------------------

def analyze_message(message):
    # -----------------------------
    # 1. ML message prediction
    # -----------------------------
    result, probability = predict_message(message)

    ml_risk = probability * 100

    # -----------------------------
    # 2. Rule-based analysis
    # -----------------------------
    warnings = detect_warning_signs(message)

    security_score = calculate_security_score(message)

    # -----------------------------
    # 3. URL analysis
    # -----------------------------
    url_findings = analyze_urls(message)

    # Dedicated phishing URL model
    url_predictions = analyze_url_predictions(message)

    # -----------------------------
    # 4. Add phishing URL risk
    # -----------------------------
    for url_result in url_predictions:
        if url_result["prediction"] == "PHISHING":
            security_score += 30

    # Keep score between 0 and 100
    security_score = min(security_score, 100)

    # -----------------------------
    # 5. Check for phishing URLs
    # -----------------------------
    has_phishing_url = any(
        url_result["prediction"] == "PHISHING"
        for url_result in url_predictions
    )

    # Check whether any URL exists
    has_url = len(url_predictions) > 0

    # -----------------------------
    # 6. Final risk assessment
    # -----------------------------
    if has_phishing_url:
        overall_assessment = "HIGH RISK"

    elif security_score >= 60:
        overall_assessment = "HIGH RISK"

    elif ml_risk >= 70 and not has_url:
        overall_assessment = "HIGH RISK"

    elif ml_risk >= 70 and has_url:
        overall_assessment = "SUSPICIOUS"

    elif security_score >= 30 or ml_risk >= 30:
        overall_assessment = "SUSPICIOUS"

    else:
        overall_assessment = "LOW RISK"

    # -----------------------------
    # 7. Return complete analysis
    # -----------------------------
    return {
        "prediction": result,
        "ml_risk_percentage": round(ml_risk, 2),
        "security_score": security_score,
        "overall_assessment": overall_assessment,
        "warning_signs": warnings,
        "url_findings": url_findings,
        "url_predictions": url_predictions
    }