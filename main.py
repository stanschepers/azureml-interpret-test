from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from interpret.ext.blackbox import TabularExplainer

breast_cancer_data = load_breast_cancer()
classes = breast_cancer_data.target_names.tolist()

x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,
                                                    breast_cancer_data.target,
                                                    test_size=0.2,
                                                    random_state=42)
clf = RandomForestClassifier()
model = clf.fit(x_train, y_train)

# "features" and "classes" fields are optional
explainer = TabularExplainer(model,
                             x_train,
                             features=breast_cancer_data.feature_names,
                             classes=classes)

# you can use the training data or the test data here
global_explanation = explainer.explain_global(x_train)

# sorted feature importance values and feature names
sorted_global_importance_values = global_explanation.get_ranked_global_values()
sorted_global_importance_names = global_explanation.get_ranked_global_names()
print(f'Type: {type(sorted_global_importance_values)}, Value: {sorted_global_importance_values}')
print(f'Type: {type(sorted_global_importance_names)}, Value: {sorted_global_importance_names}')
print(dict(zip(sorted_global_importance_names, sorted_global_importance_values)))
