from predictor import analyze_message


def run_test(message):
    print("\n" + "=" * 70)
    print("MESSAGE:")
    print(message)

    result = analyze_message(message)

    print("\nML Prediction:")
    print(result["prediction"])

    print("ML Risk:")
    print(result["ml_risk_percentage"], "%")

    print("Security Score:")
    print(result["security_score"], "/100")

    print("Overall Assessment:")
    print(result["overall_assessment"])

    print("\nWarning Signs:")

    if result["warning_signs"]:
        for warning in result["warning_signs"]:
            print("•", warning)
    else:
        print("None")

    print("\nURL Predictions:")

    if result["url_predictions"]:
        for url_result in result["url_predictions"]:
            print("URL:", url_result["url"])
            print(
                "Phishing Probability:",
                url_result["phishing_probability"],
                "%"
            )
            print(
                "Prediction:",
                url_result["prediction"]
            )
    else:
        print("None")


if __name__ == "__main__":

    test_messages = [
        "Hi, are we still meeting at college tomorrow?",
        "Congratulations! You won ₹50,000. Claim your prize at https://example.com/winner",
        "URGENT! Your bank account will be suspended. Verify your account at https://192.168.1.25/login",
        "Your OTP is 482913. Share it with customer support to complete your verification.",
        "Visit https://www.google.com for information."
    ]

    for message in test_messages:
        run_test(message)