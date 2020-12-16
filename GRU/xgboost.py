from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from h5py import File as h5file
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import xgboost as xgb
import matplotlib.pyplot  as plt
from xgboost.sklearn import XGBClassifier
import logging
from sklearn.model_selection import GridSearchCV
import time

if __name__ == '__main__':
    


    logging.getLogger('train.py').setLevel(logging.DEBUG)
    # logging.basicConfig(filename = os.path.join(os.getcwd(), 'wllog.txt'), level = logging.DEBUG) 
    logging.basicConfig(level=logging.DEBUG,
                    filename='new.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    
                    )
    # file_nameh5 = '/home/data_new/zhangyongqing/flx/pythoncode/SLP_ct2.h5'
    file_nameh5 = '/opt/data/private/wlin/SLP_ct2.h5'
    datasetfile = h5file(file_nameh5, 'r')
    X=np.array(datasetfile['/X_ct2'])
    Y=np.array(datasetfile['/Y_ct2'])

    X_train, X_test, y_train, y_test = train_test_split(X[:, :], Y, test_size=0.3, random_state=2020)
    best_acc=0
    best_depth=0
    best_rc=0
    learnrate=[0.1]
    for learn_rate in learnrate:
        for m_depth in range(33,50,10):
            start = time.clock()
            clf = XGBClassifier(
                silent=False,  
                nthread=-1,
                learning_rate=learn_rate, 
                min_child_weight=1,
                max_depth=m_depth,  
                 
                gamma=0,  
                subsample=0.9,  
                max_delta_step=0,  
                colsample_bytree=1,  
                reg_lambda=1,  
                objective= 'multi:softmax',
                
                n_estimators=1000,
                
            )
            
            # clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
            clf.fit(X_train, y_train, eval_metric="mlogloss",eval_set=[(X_train, y_train), (X_test, y_test)], verbose=True)
            
            evals_result=clf.evals_result()
            # print(evals_result)
            
            newfile = h5file('/opt/data/private/wlin/learn_rate-'+str(learn_rate)+'m_depth-'+str(m_depth)+'.h5','w')
            # newfile = h5file('/home/data_new/zhangyongqing/flx/wlin/learn_rate-'+str(learn_rate)+'m_depth-'+str(m_depth)+'.h5','w')
            newfile['/trainloss']=evals_result['validation_0']['mlogloss']
            newfile['/testloss']=evals_result['validation_1']['mlogloss']

            
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            newfile['/y_test']=y_test
            newfile['/y_pred']=y_pred
            newfile['/accuracy']=accuracy 
            # newfile['/accuracy'].attrs['train_result'] = str(classification_report(y_train, clf.predict(X_train)))
            # newfile['/accuracy'].attrs['test_result'] = str(classification_report(y_train, y_pred))
            
            print('learn_rate:',learn_rate,'m_depth:',m_depth)
            print(classification_report(y_train, clf.predict(X_train)))
            print(classification_report(y_test, y_pred))
            print('learn_rate:',learn_rate,'m_depth:',m_depth,'accuarcy:%.2f%%' % (accuracy * 100))
            newfile.close()
            import pickle
            # savemodelname='/home/data_new/zhangyongqing/flx/wlin/learn_rate-'+str(learn_rate)+'m_depth-'+str(m_depth)+'.pickle'
            savemodelname='/opt/data/private/wlin/learn_rate-'+str(learn_rate)+'m_depth-'+str(m_depth)+'.pickle'
            with open(savemodelname, 'wb') as f:
                pickle.dump(clf, f)
                pickle.dump(clf, f)

            end = time.clock()
            
            print('learn_rate:',learn_rate,'m_depth:',m_depth,'time:',end-start)
            if best_acc<accuracy:
                best_acc=accuracy
                best_depth=m_depth
                best_rc=learn_rate
    print('best_depth:',best_depth,'best_rate:',best_rc,'best_acc:',best_acc)
    # fig = plt.figure(figsize=(1000,800))
    # ax = fig.gca()
    # xgb.plot_importance(model)
    # plt.show()
    # newfile = h5py.File('/home/data_new/zhangyongqing/flx/sleepfeture/shhs1-'+str(num)+'.h5','w')
    #         newfile['/feture']=bagdata
    #         newfile['/lable']=lable
    #         newfile.close()





