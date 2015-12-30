from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import numpy as np
import operator
import re


def BasicModels(train_features, train_target, test_features, test_target):

    """
        Basic model with out using and pre-processing and use all the feature as training 
    """
    print("accuracy of different model with using any feature pre-processing")
    
    accuracy_dict = {}

    """
       KNeighborsClassifier classifier
    """
    print(" Accuracy by KNeighborsClassifier ")
    knn = KNeighborsClassifier(n_neighbors=23)
    knn.fit(train_features, train_target)
    pred = knn.predict(test_features)
    kaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["knn"]=kaccuracy
    print kaccuracy

    """
       RandomForestClassifier classifier
    """
    print(" Accuracy by RandomForestClassifier ")
    Rf = RandomForestClassifier(n_estimators=200)
    Rf.fit(train_features, train_target)
    pred = Rf.predict(test_features)
    Rfaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["RF"] = Rfaccuracy
    print Rfaccuracy

    """
       LogisticRegression classifier
    """
    print(" Accuracy by LogisticRegression ")
    LR = LogisticRegression(C=1e5, tol=0.001, fit_intercept=True)
    LR.fit(train_features, train_target)
    pred = LR.predict(test_features)
    LRaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["LR"] = LRaccuracy
    print LRaccuracy
    
    """
       DecisionTreeClassifier classifier
    """
    print(" Accuracy by DecisionTreeClassifier ")
    DT = DecisionTreeClassifier(min_samples_split=1, random_state=0)
    DT.fit(train_features, train_target)
    pred = DT.predict(test_features)
    DTaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["DT"] = DTaccuracy
    print DTaccuracy
    
    """
       MultinomialNB classifier
    """
    print(" Accuracy by MultinomialNB")
    MNB = MultinomialNB()
    MNB.fit(train_features, train_target)
    pred = MNB.predict(test_features)
    MNBaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["MNB"] = MNBaccuracy
    print MNBaccuracy

    """
       GaussianNB classifier
    """
    print(" Accuracy by GaussianNB")
    GNB =  GaussianNB()
    GNB.fit(train_features, train_target)
    pred = GNB.predict(test_features)
    GNBaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["GNB"] = GNBaccuracy
    print GNBaccuracy

    """
       SVM classifier
    """
    print(" Accuracy by SVM")
    svm =  SVC(gamma=2, C=1)
    svm.fit(train_features, train_target)
    pred = svm.predict(test_features)
    svmaccuracy =  metrics.accuracy_score(test_target, pred)
    accuracy_dict["SVM"] = svmaccuracy
    print svmaccuracy

    """
       AdaBoostClassifier classifier
    """
    print(" Accuracy by AdaBoostClassifier")
    ada = AdaBoostClassifier(n_estimators=200)
    ada.fit(train_features, train_target)
    pred = ada.predict(test_features)
    adaccuracy =  metrics.accuracy_score(test_target, pred)
    accuracy_dict["AdaBoost"] = adaccuracy
    print adaccuracy

    print("Till Now Maximum accuracy By")

    """
       Overall classifier
    """
    maximum = max(accuracy_dict, key=accuracy_dict.get)
    print(maximum, accuracy_dict[maximum])

    return maximum, accuracy_dict[maximum]



def pca(X): # X is the input data
    
    """
        After applying normalize_features pca will applied on dataset
        apply pca on both train_features and test_features
    """
    sklearn_pca = PCA(n_components=2)
    X_transf = sklearn_pca.fit_transform(X)
    return X_transf


def normalize_features_SD(train, test):
    
    """
        Normalizes train set features to a standard normal distribution
        (zero mean and unit variance). The same procedure is then applied
        to the test set features.
    """

    train_mean = train.mean(axis=0)
    # +0.1 to avoid division by zero in this specific case
    train_std = train.std(axis=0) + 0.1
    
    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    return train, test

def normalize_features(matrix_features):
    
    """
    Perform dataset normalization.
    """

    max_features = matrix_features.max(axis = 0)
    max_features = (max_features + (max_features == 0))

    return matrix_features / max_features

def Apply_Normalization(train_features, train_target, test_features, test_target):
    
    """
        Before applying pca apply some Apply_Normalization 
    """
    train_features_normal = normalize_features(train_features)
    test_features_normal = normalize_features(test_features)
    train_features_pca = pca(train_features_normal)
    test_features_pca = pca(test_features_normal)

    return (train_features_pca, train_target, test_features_pca, test_target)

def Normalizes_Featured_Model(train_features_normal, train_target, test_features_normal, test_target):

    """
        model with Normalizes feature some of the model are not work 
        as after normalize_features some data become negative
    """

    print(" Accuracy of different model with using any feature pre-processing")
    
    accuracy_dict = {}
    
    """
        KNeighborsClassifier classifier
    """
    print(" Accuracy by KNeighborsClassifier")
    knn = KNeighborsClassifier(n_neighbors=23)
    knn.fit(train_features, train_target)
    pred = knn.predict(test_features)
    kaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["knn"]=kaccuracy
    print kaccuracy
    
    """
        RandomForestClassifier classifier
    """
    print(" Accuracy by RandomForestClassifier")
    Rf = RandomForestClassifier(n_estimators=200)
    Rf.fit(train_features, train_target)
    pred = Rf.predict(test_features)
    Rfaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["RF"] = Rfaccuracy
    print Rfaccuracy
    
    """
        LogisticRegression classifier
    """
    print(" Accuracy by LogisticRegression")
    LR = LogisticRegression(C=1e5, tol=0.001, fit_intercept=True)
    LR.fit(train_features, train_target)
    pred = LR.predict(test_features)
    LRaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["LR"] = LRaccuracy
    print LRaccuracy
    
    """
        DecisionTreeClassifier classifier
    """
    print(" Accuracy by DecisionTreeClassifier")
    DT = DecisionTreeClassifier(min_samples_split=1, random_state=0)
    DT.fit(train_features, train_target)
    pred = DT.predict(test_features)
    DTaccuracy = metrics.accuracy_score(test_target, pred)
    accuracy_dict["DT"] = DTaccuracy
    print DTaccuracy
    
    """
        SVM classifier
    """
    print(" Accuracy by SVM")
    svm =  SVC(gamma=2, C=1)
    svm.fit(train_features, train_target)
    pred = svm.predict(test_features)
    svmaccuracy =  metrics.accuracy_score(test_target, pred)
    accuracy_dict["SVM"] = svmaccuracy
    print svmaccuracy

    print("Till Now Maximum accuracy By")
    
    """
        Overall Accuracy
    """
    maximum = max(accuracy_dict, key=accuracy_dict.get)
    print(maximum, accuracy_dict[maximum])
    return maximum, accuracy_dict[maximum]


def PreprocessData(train_features,test_features):
    
    """
        Apply some preprocessing
    """
    scaler = preprocessing.StandardScaler().fit(train_features)
    Transformed_feature = scaler.transform(train_features)
    X_test_transformed = scaler.transform(test_features)
    return (Transformed_feature,X_test_transformed)

def Model_After_Preprocessing(train_features,train_target,test_features,test_target):
    
    """
        Model after applying some preprocessing steps
    """
    Transformed_train_feature,Transformed_test_feature = PreprocessData(train_features,test_features)
    PreprocessData_accurary = {}
    
    """
        SVM model accuracy
    """
    print(" SVM accuracy")
    clf = SVC(C=1).fit(Transformed_train_feature, train_target)
    svmaccuracy = clf.score(Transformed_test_feature, test_target)
    PreprocessData_accurary["SVM"] = svmaccuracy
    print svmaccuracy

    """
        KNeighborsClassifier classification Model
    """

    print(" Accuracy by KNeighborsClassifier")
    knn = KNeighborsClassifier(n_neighbors=23).fit(Transformed_train_feature, train_target)
    knnaccuracy = knn.score(Transformed_test_feature,test_target)
    PreprocessData_accurary["knn"] = knnaccuracy
    print knnaccuracy

    """
        RandomForestClassifier classifier
    """
    print(" Accuracy by RandomForestClassifier")
    Rf = RandomForestClassifier(n_estimators=200).fit(Transformed_train_feature, train_target)
    RFAccuracy = Rf.score(Transformed_test_feature,test_target)
    PreprocessData_accurary["RF"] = RFAccuracy
    print RFAccuracy

    """
        LogisticRegression classifier
    """
    print(" Accuracy by LogisticRegression")
    LR = LogisticRegression(C=1e5, tol=0.001, fit_intercept=True).fit(Transformed_train_feature, train_target)
    LRaccuracy = LR.score(Transformed_test_feature,test_target)
    PreprocessData_accurary["LR"] = LRaccuracy
    print LRaccuracy

    """
        DecisionTreeClassifier classifier
    """
    print(" Accuracy by DecisionTreeClassifier")
    DT = DecisionTreeClassifier(min_samples_split=1, random_state=0).fit(Transformed_train_feature, train_target)
    DTaccuracy = DT.score(Transformed_test_feature,test_target)
    PreprocessData_accurary["DT"] = DTaccuracy
    print DTaccuracy


    """
        AdaBoostClassifier classifier
    """
    print(" Accuracy by AdaBoostClassifier")
    ada = AdaBoostClassifier(n_estimators=200).fit(Transformed_train_feature, train_target)
    adaccuracy = ada.score(Transformed_test_feature,test_target)
    PreprocessData_accurary["ada"] = adaccuracy
    print adaccuracy

    """
        Maximum accuracy after applying some preprocessing steps
    """
    maximum = max(PreprocessData_accurary, key=PreprocessData_accurary.get)
    print(maximum, PreprocessData_accurary[maximum])
    return maximum, PreprocessData_accurary[maximum]


def Use_classification_Model(train_features, train_target, test_features, test_target):

    print(" use basic model with all the features as training ")

    """
        All the feature are select after applying 
    """
    accuracy_dict = {}

    model1, basic_accuracy = BasicModels(train_features, train_target, test_features, test_target)
    accuracy_dict[model1] = basic_accuracy

    # raw_input(" Press Enter to apply some feature extration... ")

    train_features_normal, train_target, test_features_normal, test_target = Apply_Normalization(train_features, 
                                                                                                 train_target,
                                                                                                 test_features, 
                                                                                                 test_target)

    
    model2, normal_accuracy = Normalizes_Featured_Model(train_features_normal, 
                                                        train_target, 
                                                        test_features_normal, 
                                                        test_target)
                                                        
    accuracy_dict[model2] = normal_accuracy

    """ After applying Apply_Normalization on the train_features and test_features some of the value become 
        negative so few model will not work 
    """
    # raw_input("Apply some preprocessing ...")

    process_feature_train, process_feature_test = PreprocessData(train_features,test_features)


    """
        Apply some cross validation on the model and check, which are the parameters that can be apply to fitted well 
        for applying closs validation we are using all train_features and whole target features
        apply cross validation only on the train_features and train_target
    """
    model3, prepross_accuracy = Model_After_Preprocessing(train_features,train_target,test_features,test_target)
    accuracy_dict[model3] = prepross_accuracy

    
    # raw_input(" Press Enter to apply features selection on unnormalize features ...")

    print(" include some of the feature and exclude some of the feature ")

    """ select some of the column number of features deselect some of the feature 
        here we are taking some of the variable that are selecting and some of the variable
        that are use for excluding
    """

    SELECTED_FEATURES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]

    """
        All the feature are select after applying 
    """
    
    train_feature = train_features[SELECTED_FEATURES]
    test_feature = test_features[SELECTED_FEATURES]

    model4, feature_accuracy = BasicModels(train_features, train_target, test_features, test_target)
    accuracy_dict[model4] = feature_accuracy
   
    """
        Apply_Normalization on the selected features
    """ 
    # raw_input("apply some Apply_Normalization on the selected features")
    train_features_normal, train_target, test_features_normal, test_target = Apply_Normalization(train_feature, 
                                                                                                 train_target,
                                                                                                 test_feature, 
                                                                                                 test_target)
    
    model5, norml_feature_accuracy = Normalizes_Featured_Model(train_features_normal, train_target, test_features_normal, test_target)
    accuracy_dict[model5] = norml_feature_accuracy

    print("maximum accuracy by all model")
    print accuracy_dict


def extract(filename):
   
    """
        Basic pre-processing logic to load dataset information.
    """
    INPUT_FILE = open(filename)
    DATA = INPUT_FILE.readlines()
    FEATURES = []
    TARGET = []

    for line in DATA:
        formatted_line = line.strip("\n")
        TARGET_I = formatted_line.split(" ")[1]
        FEATURES_I = re.sub(r"(\d+):", "", formatted_line).split(" ")[2:]

        TARGET.append(TARGET_I)
        FEATURES.append(FEATURES_I)

    FEATURES = np.array(FEATURES).astype(np.float)
    TARGET = np.array(TARGET).astype(np.int)

    INPUT_FILE.close()

    return FEATURES, TARGET

def dataset_load(train_filename, test_filename):

    """
        Experiment considering raw dataset (no modification).
    """
    (train_features, train_targets) = extract(train_filename)
    (test_features, test_targets)   =   extract(test_filename)
    return train_features, train_targets, test_features, test_targets

    """
        Now wee have features and targets for both training data and test dataset use this for model training.
    """

if __name__ == "__main__":
    train_dataset = 'dataset/train.txt'
    test_dataset = 'dataset/test.txt'
    (train_features, train_target, test_features, test_target) = dataset_load(train_dataset, test_dataset)
    Use_classification_Model(train_features, train_target, test_features, test_target)