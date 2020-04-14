from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import train_test_split
from interpret.ext.blackbox import PFIExplainer, TabularExplainer



breast_cancer_data = load_breast_cancer()
classes = breast_cancer_data.target_names.tolist()

# split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,            
                                                    breast_cancer_data.target,  
                                                    test_size=0.2,
                                                    random_state=0)
clf = svm.SVC(gamma=0.001, C=100., probability=True)
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
dict(zip(sorted_global_importance_names, sorted_global_importance_values))
