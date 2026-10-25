from classification.request_type import classify_request_type
from routing.decision_engine import decide_status


def run() -> None:
    assert classify_request_type("Feature request for dark mode", "") == "feature_request"
    assert classify_request_type("App is down", "") == "bug"
    assert classify_request_type("Which actor won", "") == "invalid"
    assert decide_status("invalid", "", [], 10.0) == "replied"
    assert decide_status("product_issue", "", [], 10.0) == "escalated"


if __name__ == "__main__":
    run()
    print("sanity tests passed")
