from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, check_scoring
import numpy as np

def test_check_scoring_with_scorers():
    acc_scorer = make_scorer(accuracy_score)
    roc_scorer = make_scorer(roc_auc_score, needs_threshold=True)
    
    scoring_dict = {
        "accuracy": acc_scorer,
        "roc_auc": roc_scorer
    }
    
    # This is what BaseParadigm does
    print("Pre-checks with None estimator...")
    try:
        check_scoring(None, scoring=scoring_dict)
        print("Pre-check passed")
    except Exception as e:
        print(f"Pre-check failed: {e}")

    class MockEstimator:
        def __init__(self):
            self._estimator_type = "classifier"
            self.classes_ = np.array([0, 1])
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.array([0, 1])
        def decision_function(self, X):
            return np.array([0.1, 0.9])
            
    print("Calling check_scoring with estimator...")
    scorer = check_scoring(MockEstimator(), scoring=scoring_dict)
    print(f"Resulting scorer: {scorer}")
    
    y_true = np.array([0, 1])
    try:
        print("Calling scorer...")
        val = scorer(MockEstimator(), None, y_true)
        print(f"Values: {val}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_check_scoring_with_scorers()
