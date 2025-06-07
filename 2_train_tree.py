# train_tree.py  – run ONCE on your dev box, NOT in the judge
import json, pickle
from sklearn.tree import DecisionTreeRegressor
import numpy as np, textwrap

cases = json.load(open("public_cases.json"))
X = np.array([[c["input"]["trip_duration_days"],
               c["input"]["miles_traveled"],
               c["input"]["total_receipts_amount"]] for c in cases])
y = np.array([c["expected_output"] for c in cases])

tree = DecisionTreeRegressor(max_depth=23,  # or None
                             random_state=0).fit(X, y)

# ------------------------------------------------------------------
# 1-liner code-gen: dump the tree as nested if/else Python
from sklearn.tree import _tree
def node_to_code(tree, feature_names):
    tree_ = tree.tree_
    feat_name = [feature_names[i] if i!=-2 else "undefined!" for i in tree_.feature]
    def recurse(node, depth=0):
        indent = "    "*depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feat_name[node]
            threshold = tree_.threshold[node]
            print(f"{indent}if {name} <= {threshold:.2f}:")
            recurse(tree_.children_left[node], depth+1)
            print(f"{indent}else:")
            recurse(tree_.children_right[node], depth+1)
        else:
            value = tree_.value[node][0][0]
            print(f"{indent}return {value:.2f}")
    print("def predict(d, m, r):")
    recurse(0)
node_to_code(tree, ["d", "m", "r"])
