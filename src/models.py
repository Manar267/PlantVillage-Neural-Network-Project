from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"[{model_name}] Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    return {"model": model_name, "accuracy": acc, "macro_f1": macro_f1}


def run_trivial_baseline(X_train, y_train, X_val, y_val, strategy="most_frequent"):
    clf = DummyClassifier(strategy=strategy, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    return evaluate_predictions(y_val, preds, model_name=f"Trivial ({strategy})"), clf


def run_simple_learned_baseline(X_train, y_train, X_val, y_val, max_iter=1000):
    clf = LogisticRegression(max_iter=max_iter, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    return evaluate_predictions(y_val, preds, model_name="Simple Learned (LogReg)"), clf
