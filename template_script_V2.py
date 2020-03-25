"""
    Python script to submit as a part of the project of ELTP 2020 course.
    
    This script serves as a template. Please use proper comments and meaningful variable names.
"""

"""
    Group Members:
        (1) VASHISHT Raghav B00766425 raghav.vashisht@student-cs.fr
        (2) KOGIAS Kleomenis B00763236 kleomenis.kogias@student-cs.fr  
        (3) WANG Xiangyu B00759895 xiangyu.wang@student-cs.fr
        
    --- Submission on the competition website by the name : ------
"""

"""
    Import necessary packages - NOTE - Here we are simply importing all 
    the required packages, as we assume that all the packages have already
    been installed including the french package for Spacy
"""
import numpy as np
import pandas as pd
import re
import spacy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble

from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

spacy_nlp = spacy.load('fr')

"""
Step - importing the train file (along with the labels) and test file 
"""
# Import training and testing data 
train = pd.read_csv('D:\\All Essec\\DS electives\\Ratuken Project\\data\\X_train_update.csv', index_col=0)

test = pd.read_csv('D:\\All Essec\\DS electives\\Ratuken Project\\data\\X_test_update.csv', index_col=0)

y_train = pd.read_csv('D:\\All Essec\\DS electives\\Ratuken Project\\data\\Y_train_CVw08PX.csv', index_col=0)

# dropping the irrelevant columns from the data frames

del train['description'], train['productid'], train['imageid']

del test['description'], test['productid'], test['imageid']

#renaming the column in y train 
y_train = y_train.rename(columns={"prdtypecode": "label"})

"""
Step - Precprocessing - 
Here we will create the required TF - IDF matrix usnig the following pre processing steps:
    
    1. Lower case the strings
    2. Handling Accented Character
    3. Tokenization
    4. Handling Punctuations
    5. Stop word filtering
    6. Removing  numbers and special symbols
    
"""
def normalize_accent(string):
    string = string.replace('á', 'a')
    string = string.replace('â', 'a')

    string = string.replace('é', 'e')
    string = string.replace('è', 'e')
    string = string.replace('ê', 'e')
    string = string.replace('ë', 'e')

    string = string.replace('î', 'i')
    string = string.replace('ï', 'i')

    string = string.replace('ö', 'o')
    string = string.replace('ô', 'o')
    string = string.replace('ò', 'o')
    string = string.replace('ó', 'o')

    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('ü', 'u')

    string = string.replace('ç', 'c')
    
    return string

# function to tokenize
def raw_to_tokens(raw_string, spacy_nlp):
    # Write code for lower-casing
    string = raw_string.lower()
    
    # Write code to normalize the accents
    string = normalize_accent(string)
        
    # Write code to tokenize
    spacy_tokens = spacy_nlp(string)
        
    # Write code to remove punctuation tokens and create string tokens
    string_tokens = [token.orth_ for token in spacy_tokens if not token.is_punct if not token.is_stop]
    
    # Write code to join the tokens back into a single string
    clean_string = " ".join(string_tokens)
    
    return clean_string


# remove the special char like degree or phi 
train['designation_sp'] = train['designation'].apply(lambda x: re.sub('\W+'," ", x ))
test['designation_sp'] = test['designation'].apply(lambda x: re.sub('\W+'," ", x ))

# reomve numbers
train['designation_number'] = train['designation_sp'].str.replace('\d+', '')
test['designation_number'] = test['designation_sp'].str.replace('\d+', '')

docs_train = train['designation_number']


# tokenize the train file
train['designation_token'] = train['designation_number'].apply(lambda x: raw_to_tokens(x, spacy_nlp))

# tokenize the test file
test['designation_token'] = test['designation_number'].apply(lambda x: raw_to_tokens(x, spacy_nlp))


"""
Step - Creating the TF - IDF Matrix and hence crating the test and train datasets
"""

# makes list for the train and test files

desc_list_train = train['designation_token']
desc_list_test = test['designation_token']

frames = [desc_list_train, desc_list_test]
consolidated_desc = pd.concat(frames)

# Write code to import TfidfVectorizer

# Write code to create a TfidfVectorizer object
tfidf = TfidfVectorizer()

# Write code to vectorize the sample text
X_tfidf_sample = tfidf.fit_transform(consolidated_desc.astype('U'))

print("Shape of the TF-IDF Matrix:")
print(X_tfidf_sample.shape)


X = X_tfidf_sample[:84916, : ]
y = y_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
"""
    Your methods implementing the models.
    
    Each of your model should have a separate method. e.g. run_random_forest, run_decision_tree etc.
    
    Your method should:
        (1) create the proper instance of the model with the best hyperparameters you found
        (2) fit the model with a given training data
        (3) run the prediction on a given test data
        (4) return accuracy and F1 score
        
    Following is a sample method. Please note that the parameters given here are just examples.
    
    --------NOTE - we are using a random search search with cross validation for
        hyperparameters tuning, hence defining the test train split is redundant---------
"""

"""
MODEL - 1 RANDOM FOREST
"""
def model_random_forest(X_train, X_test, y_train, y_test):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(50, 250, num = 20)]
    
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
    max_depth.append(None)
    
    # Minimum number of samples required to split a node
    min_samples_split = [1, 2, 5, 10, 25]
    
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 8, 16]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                    n_iter = 30, cv =10, verbose=1, random_state=42, scoring = 'f1_weighted', n_jobs=-1)
    
    # Fit the random search model
    rf_random.fit(X_train, y_train.values.ravel())
    
    optimised_random_forest = rf_random.best_estimator_

    y_predicted = optimised_random_forest.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_predicted)
    rf_f1 = f1_score(y_test, y_predicted, average="weighted")

    return rf_accuracy, rf_f1

"""
MODEL - 2 GRADIENT BOOSTING CLASSIFIER
"""
def model_gradient_boost(X_train, X_test, y_train, y_test):

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(50, 250, num = 20)]
    
    # Learning rate of the boosting
    learning_rate =  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ]
    
    # Max number of leaf nodes
    max_leaf_nodes = [int(x) for x in np.linspace(2, 25, num = 10)]
    
    # Minimum number of samples required at each leaf node
    max_depth = [int(x) for x in np.linspace(3, 50, num = 10)]
    
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate,
                   'max_leaf_nodes': max_leaf_nodes,
                   'max_depth': max_depth}
    


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    gb = ensemble.GradientBoostingClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    gb_random = RandomizedSearchCV(estimator = gb, param_distributions = random_grid,
                    n_iter = 30, cv =10, verbose=1, random_state=42, scoring = 'f1_weighted', n_jobs=-1)
    
    # Fit the random search model
    gb_random.fit(X_train, y_train.values.ravel())
    
    optimised_gradient_boosting = gb_random.best_estimator_

    y_predicted = optimised_gradient_boosting.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_predicted)
    gb_f1 = f1_score(y_test, y_predicted, average="weighted")

    return gb_accuracy, gb_f1


"""
MODEL - 3 XG BOOST
"""
def model_XGboost(X_train, X_test, y_train, y_test):

    
    random_grid = {
     "n_estimators"     : [int(x) for x in np.linspace(50, 500, num = 20)],
     "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    xgb_model = XGBClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    xgb_random = RandomizedSearchCV(estimator = xgb_model, param_distributions = random_grid,
    n_iter = 30, cv = 10, verbose=2, random_state=42 ,n_jobs = -1, scoring = 'f1_weighted')
    
    # Fit the random search model
    xgb_random.fit(X_train, y_train.values.ravel())
    
    optimised_xgb_random = xgb_random.best_estimator_

    y_predicted = optimised_xgb_random.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, y_predicted)
    xgb_f1 = f1_score(y_test, y_predicted, average="weighted")

    return xgb_accuracy, xgb_f1


"""
MODEL - 4 Decision Tree Classifier
"""
def model_DecisionTreeClassifier(X_train, X_test, y_train, y_test):

    
    random_grid = {
     "criterion"     : ['gini', 'entropy'],
     "max_depth"     : [4, 6, 8, 12, 20, 40,50] ,
     "splitter"      : ["best", "random"]}


    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    clf_dt = DecisionTreeClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    dt_random = RandomizedSearchCV(estimator = clf_dt, param_distributions = random_grid,
    n_iter = 30, cv = 10, verbose=2, random_state=42 ,n_jobs = -1, scoring = 'f1_weighted')
    
    # Fit the random search model
    dt_random.fit(X_train, y_train.values.ravel())
    
    optimised_dt_random = dt_random.best_estimator_

    y_predicted = optimised_dt_random.predict(X_test)
    dt_accuracy = accuracy_score(y_test, y_predicted)
    dt_f1 = f1_score(y_test, y_predicted, average="weighted")

    return dt_accuracy, dt_f1


"""
MODEL - 5 ADA Boost Classifier
"""
def model_ADABoost(X_train, X_test, y_train, y_test):

    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(30, 250, num = 20)]
    
    # Learning rate
    learning_rate = [x for x in np.linspace(0.001, 2.5, num = 20)] 

    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    ada = AdaBoostClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    ada_random = RandomizedSearchCV(estimator = ada, param_distributions = random_grid,
                    n_iter = 20, cv =10, verbose=-1, random_state=42, scoring = 'f1_weighted', n_jobs=-1)
    
    # Fit the random search model
    ada_random.fit(X_train, y_train.values.ravel())
    
    optimised_ada_boost = ada_random.best_estimator_

    y_predicted = optimised_ada_boost.predict(X_test)
    ada_accuracy = accuracy_score(y_test, y_predicted)
    ada_f1 = f1_score(y_test, y_predicted, average="weighted")

    return ada_accuracy, ada_f1


"""
MODEL - 6 Bagging Classifier
"""
def model_BaggingClassifier(X_train, X_test, y_train, y_test):

    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(10, 150, num = 20)]
    
    # The number of samples to draw from X to train each base estimator.
    max_samples = [1, 2, 5, 10, 25]
    
    # The number of features to draw from X to train each base estimator.
    max_features = [1, 2, 4, 8, 16]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_samples': max_samples,
                   'max_features': max_features,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    bag = BaggingClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    bag_random = RandomizedSearchCV(estimator = bag, param_distributions = random_grid,
                    n_iter = 50, cv =10, verbose=-1, random_state=42, scoring = 'f1_weighted', n_jobs=-1)
    
    # Fit the random search model
    bag_random.fit(X_train, y_train.values.ravel())
    
    optimised_bag = bag_random.best_estimator_

    y_predicted = optimised_bag.predict(X_test)
    bag_accuracy = accuracy_score(y_test, y_predicted)
    bag_f1 = f1_score(y_test, y_predicted, average="weighted")

    return bag_accuracy, bag_f1

"""

**************VERY IMPORTANT NOTE REGARDING THE RESULTS******************

PLease be advised that the main function in this script produced the results based on
 the X_train, X_test, y_train, y_test values obtianed form the test- train split. 

While the numbers obtained in the report are from the Rakuten Data Challange website
enterd by uploadein the predictions on the X_test file. 

Hence, comparing the two F1 scores and accuracies though tempting, is a futile
exerise as they are bound to be different.

**************************************************************************
"""

"""
   The main function should print all the accuracies and F1 scores for all the models.
   
   The names of the models should be sklearn classnames, e.g. DecisionTreeClassifier, RandomForestClassifier etc.
   
   Please make sure that your code is outputting the performances in proper format, because your script will be run automatically by a meta-script.
"""
if __name__ == "__main__":
    """
       This is just an example, plese change as necceary. Just maintain final output format with proper names of the models as described above.
    """
    random_forest_acc, random_forest_f1 = model_random_forest(X_train, X_test, y_train, y_test)
    
    gradient_boosting_acc, gradient_boosting_f1 = model_gradient_boost(X_train, X_test, y_train, y_test)
    
    Extreeme_gradient_boosting_acc, Extreeme_gradient_boosting_f1 = model_XGboost(X_train, X_test, y_train, y_test)
    
    DecisionTreeClassifier_acc, DecisionTreeClassifier_f1 = model_DecisionTreeClassifier(X_train, X_test, y_train, y_test)
    
    ADA_acc, ADA_f1 = model_ADABoost(X_train, X_test, y_train, y_test)
    
    bagging_acc, bagging_f1 = model_BaggingClassifier(X_train, X_test, y_train, y_test)
    
    
    """
        etc.
    """

    # print the results
    print("Random Forest", random_forest_acc, random_forest_f1)
    print("Gradient Boosting", gradient_boosting_acc, gradient_boosting_f1)
    print("XGBoost", Extreeme_gradient_boosting_acc, Extreeme_gradient_boosting_f1)
    print("Decision Tree Classifier", DecisionTreeClassifier_acc, DecisionTreeClassifier_f1)
    print("ADA Boost Classifier", ADA_acc, ADA_f1)
    print("Bagging Classifier", bagging_acc, bagging_f1)
    """
        etc.
    """
