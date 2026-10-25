"""Logistic regression with the standard library only.

Training is full-batch gradient descent from a fixed start, so the same
rows always produce the same weights. A wrong reply is weighted five times
an unnecessary escalation while the weights are fit. The decision cutoff is
the lowest probability given to a training reply, so the fit does not
escalate a ticket those labels call a reply.
"""

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from prediction.features import FEATURE_NAMES


WRONG_REPLY_COST = 5
UNNECESSARY_ESCALATION_COST = 1


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def decision_cost(label: str, predicted: str) -> int:
    if label == "escalated" and predicted == "replied":
        return WRONG_REPLY_COST
    if label == "replied" and predicted == "escalated":
        return UNNECESSARY_ESCALATION_COST
    return 0


class DecisionModel:
    def __init__(
        self,
        weights: Sequence[float],
        bias: float,
        mean: Sequence[float],
        scale: Sequence[float],
        threshold: float,
    ):
        self.weights = [float(value) for value in weights]
        self.bias = float(bias)
        self.mean = [float(value) for value in mean]
        self.scale = [float(value) if float(value) else 1.0 for value in scale]
        self.threshold = float(threshold)

    def predict_proba(self, features: Dict[str, float]) -> float:
        total = self.bias
        for index, name in enumerate(FEATURE_NAMES):
            centered = (float(features[name]) - self.mean[index]) / self.scale[index]
            total += self.weights[index] * centered
        return sigmoid(total)

    def predict_status(self, features: Dict[str, float]) -> str:
        if self.predict_proba(features) >= self.threshold:
            return "replied"
        return "escalated"

    def to_dict(self) -> dict:
        return {
            "features": list(FEATURE_NAMES),
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "scale": self.scale,
            "threshold": self.threshold,
            "seed": 0,
            "wrong_reply_cost": WRONG_REPLY_COST,
            "unnecessary_escalation_cost": UNNECESSARY_ESCALATION_COST,
        }


def _columns(rows: Sequence[Dict[str, float]]) -> Tuple[List[float], List[float], List[List[float]]]:
    mean = []
    scale = []
    columns: List[List[float]] = []
    for name in FEATURE_NAMES:
        values = [float(row[name]) for row in rows]
        columns.append(values)
        center = sum(values) / len(values)
        variance = sum((value - center) ** 2 for value in values) / len(values)
        mean.append(center)
        scale.append(math.sqrt(variance) or 1.0)
    return mean, scale, columns


def train_model(
    rows: Sequence[Dict[str, float]],
    labels: Sequence[str],
    steps: int = 800,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> DecisionModel:
    if len(rows) != len(labels) or not rows:
        raise ValueError("training rows and labels must be the same non-empty length")
    mean, scale, columns = _columns(rows)
    coded = []
    sample_weight = []
    for label in labels:
        if label not in ("replied", "escalated"):
            raise ValueError(f"unknown label {label}")
        coded.append(1.0 if label == "replied" else 0.0)
        sample_weight.append(1.0 if label == "replied" else float(WRONG_REPLY_COST))
    standardized = []
    for row_index in range(len(rows)):
        standardized.append(
            [(columns[feature][row_index] - mean[feature]) / scale[feature] for feature in range(len(FEATURE_NAMES))]
        )
    weights = [0.0] * len(FEATURE_NAMES)
    bias = 0.0
    count = float(len(rows))
    for _step in range(steps):
        grad_w = [0.0] * len(FEATURE_NAMES)
        grad_b = 0.0
        for features, label, weight in zip(standardized, coded, sample_weight):
            score = bias + sum(coeff * value for coeff, value in zip(weights, features))
            error = weight * (sigmoid(score) - label)
            grad_b += error
            for index, value in enumerate(features):
                grad_w[index] += error * value
        bias -= learning_rate * (grad_b / count)
        for index in range(len(weights)):
            weights[index] -= learning_rate * ((grad_w[index] / count) + l2 * weights[index])
    model = DecisionModel(weights, bias, mean, scale, threshold=0.5)
    reply_probabilities = [
        model.predict_proba(features)
        for features, label in zip(rows, labels)
        if label == "replied"
    ]
    # The loss already charges a wrong reply five times an unnecessary
    # escalation. The cutoff is the lowest probability of a training reply,
    # so that fit does not escalate a ticket the training labels call a reply.
    model.threshold = min(reply_probabilities) if reply_probabilities else 0.5
    return model


def model_path() -> Path:
    return Path(__file__).resolve().parent / "decision_model.json"


def save_model(model: DecisionModel, path: Path = None) -> None:
    destination = path or model_path()
    destination.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")


def load_model(path: Path = None) -> DecisionModel:
    payload = json.loads((path or model_path()).read_text(encoding="utf-8"))
    if list(payload.get("features", [])) != list(FEATURE_NAMES):
        raise ValueError("decision model features do not match the pipeline")
    return DecisionModel(
        payload["weights"],
        payload["bias"],
        payload["mean"],
        payload["scale"],
        payload["threshold"],
    )


def precision_report(labels: Iterable[str], predicted: Iterable[str]) -> dict:
    pairs = list(zip(labels, predicted))
    wrong_replies = sum(1 for label, guess in pairs if label == "escalated" and guess == "replied")
    unnecessary = sum(1 for label, guess in pairs if label == "replied" and guess == "escalated")
    reply_calls = [label for label, guess in pairs if guess == "replied"]
    escalate_calls = [label for label, guess in pairs if guess == "escalated"]
    reply_precision = (
        sum(1 for label in reply_calls if label == "replied") / len(reply_calls) if reply_calls else 1.0
    )
    escalation_precision = (
        sum(1 for label in escalate_calls if label == "escalated") / len(escalate_calls)
        if escalate_calls
        else 1.0
    )
    return {
        "tickets": len(pairs),
        "wrong_replies": wrong_replies,
        "unnecessary_escalations": unnecessary,
        "reply_precision": reply_precision,
        "escalation_precision": escalation_precision,
        "cost": wrong_replies * WRONG_REPLY_COST + unnecessary * UNNECESSARY_ESCALATION_COST,
        "note": (
            "A linear model with these features can copy the action-overlap rule. "
            "It does not, by itself, recognize an off-topic question that shares "
            "two ordinary words with a document."
        ),
    }
