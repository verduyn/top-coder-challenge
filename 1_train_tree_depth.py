import json, numpy as np
from sklearn.tree import DecisionTreeRegressor

cases = json.load(open("public_cases.json"))
X = np.array([[c["input"]["trip_duration_days"],
               c["input"]["miles_traveled"],
               c["input"]["total_receipts_amount"]] for c in cases])
y = np.array([c["expected_output"] for c in cases])

for d in range(3, 23):
    tree = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
    mae  = np.abs(tree.predict(X) - y).mean()
    print(f"depth {d:2}:  MAE = {mae:8.2f}")
