from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pdb



def svm_classify(train_image_feats, train_labels, test_image_feats):
    #################################################################################
    # TODO :                                                                        #
    # This function will train a set of linear SVMs for multi-class classification  #
    # and then use the learned linear classifiers to predict the category of        #
    # every test image.                                                             # 
    #################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # LinearSVC :                                                                    #
    #   The multi-class svm classifier                                               #
    #        e.g. LinearSVC(C= ? , class_weight=None, dual=True, fit_intercept=True, #
    #                intercept_scaling=1, loss='squared_hinge', max_iter= ?,         #
    #                multi_class= ?, penalty='l2', random_state=0, tol= ?,           #
    #                verbose=0)                                                      #
    #                                                                                #
    #             C is the penalty term of svm classifier, your performance is highly#
    #          sensitive to the value of C.                                          #
    #   Train the classifier                                                         #
    #        e.g. classifier.fit(? , ?)                                              #
    #   Predict the results                                                          #
    #        e.g. classifier.predict( ? )                                            #
    ##################################################################################
    '''
    Input : http://localhost:8888/edit/Desktop/homework3/code/svm_classify.py#
        train_image_feats : training images features
        train_labels : training images labels
        test_image_feats : testing images features
    Output :
        Predict labels : a list of predict labels of testing images (Dtype = String).
    '''
    #svc = LinearSVC(random_state = 0)
    svc = SVC(random_state=0)
    param_C = [0.001 , 0.01 , 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    param_gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    param_grid = [{'C': param_C,
                   'gamma': param_gamma,
                   'kernel': ['rbf']}]

    gs = GridSearchCV(estimator = svc,
                      param_grid= param_grid,
                      scoring='accuracy',
                     )
    
    gs = gs.fit(train_image_feats, train_labels)
    
    print(f'Best Training Score = {gs.best_score_:.3f} with parameters {gs.best_params_}')
    
    classifier = gs.best_estimator_
    classifier.fit(train_image_feats, train_labels)
    
    
    pred_label = classifier.predict(test_image_feats)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return pred_label
