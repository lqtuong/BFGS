from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.contrib.learn.python.learn.datasets import base

from sklearn import tree
import tensorflow as tf
import numpy as np
import pandas as pd
import csv, graphviz

tf.logging.set_verbosity(tf.logging.INFO)

CHURN_TRAINING = 'data_train_dnn1.csv'
CHURN_TESTING = 'data_test_dnn1.csv'
CHURN_PREDICT = 'data_to_train_dnn2.csv'
feature_name = "churn_features"

def standard_data(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    for col in df.columns:
       if df[col].dtype != 'object':
          maxrow = float(df[col].max())
          minrow = float(df[col].min())
          for i in range(df.shape[0]):
              value = float(df.ix[i,col])
              df.ix[i,col] = (value-minrow)/(maxrow-minrow)
    df.to_csv('metadata-standard.csv')
       
def process2csv(text, csv_text):
    write = csv.writer(open(csv_text, 'w'))
    readline = csv.reader(open(text, 'r'), delimiter=',')
    write.writerows(readline)

def preprocess(inputs, method):
    df = pd.read_csv(inputs)
    if method == 'dnn1':
       df.drop(['Phone', 'Area Code', 'State'], axis=1, inplace=True)
       mapping = {'no': 0., 'yes':1., 'False.':0, 'True.':1}
       df.replace({'International Plan' : mapping, 'VMail Plan' : mapping, 'Churn':mapping}, regex=True, inplace=True)
       data_predict = df.rename(columns={'Account Length': str(df.shape[0]), 'International Plan': '17'})
       data_predict.to_csv('data_to_train_dnn2.csv', index=False)

    length = df.shape[0]
    data_train, data_test = df.ix[:int(9*length/10), :], df.ix[int(9*length/10+1):, :]
    data_train.rename(columns={'Account Length': str(data_train.shape[0]), 'International Plan': '17'}, inplace=True)
    data_test.rename(columns={'Account Length': str(data_test.shape[0]), 'International Plan': '17'}, inplace=True)

    if method == 'dnn1':
       data_train.to_csv('data_train_dnn1.csv', index=False)
       data_test.to_csv('data_test_dnn1.csv', index=False)
    if method == 'dnn2':
       data_train.to_csv('data_train_dnn2.csv', index=False)
       data_test.to_csv('data_test_dnn2.csv', index=False)
    if method == 'tree':
       data_train, data_predict = df.ix[:int(8.5*length/10),:], df.ix[int(8.5*length/10+1):int(9*length/10), :]
       data_train.to_csv('data_train_tree.csv', index=False, header=False)
       data_test.to_csv('data_test_tree.csv', index=False, header=False)    
       data_predict.to_csv('data_predict_tree.csv', index=False, header=False)

def model(mdir):
    feature_columns = [tf.feature_column.numeric_column(feature_name,
                                                        shape=[17])]
    classifier = tf.estimator.DNNClassifier(
          feature_columns=feature_columns,
          hidden_units=[10, 5, 10],
          n_classes=2,
          #optimizer='Adam',
          #activation_fn=tf.nn.softmax,
          #dropout=0.5,
          model_dir= mdir
    )
    return classifier

def input_fn(dataset):
    def _fn():
        features = {feature_name: tf.constant(dataset.data)}
        label = tf.constant(dataset.target)
        return features, label
    return _fn



def DNN1():
    preprocess('metadata-standard.csv', 'dnn1')

    training_set = base.load_csv_with_header(filename=CHURN_TRAINING,
                                             features_dtype=np.float32,
                                             target_dtype=np.int)
    test_set = base.load_csv_with_header(filename=CHURN_TESTING,
                                         features_dtype=np.float32,
                                         target_dtype=np.int)
    predict_set = base.load_csv_with_header(filename=CHURN_PREDICT,
                                         features_dtype=np.float32,
                                     target_dtype=np.int)
    classifier = model('/tmp/churn')
    classifier.train(input_fn=input_fn(training_set),steps=1000000)
    print('fit done')

    accuracy_score = classifier.evaluate(input_fn=input_fn(test_set),
                                     steps=10)["accuracy"]
    print('\nAccuracy: {0:f}'.format(accuracy_score))

    expected = predict_set
    predictions = classifier.predict(input_fn=input_fn(expected))
    print(type(predictions))
    with open('out_pre.txt', 'w') as fout:
        fout.writelines('Account Length,International Plan,VMail Plan,VMail Message,Day Mins,Day Calls,Day Charge,Eve Mins,Eve Calls,Eve Charge,Night Mins,Night Calls,Night Charge,Intl Mins,Intl Calls,Intl Charge,CustServ Calls,Churn\n')
        count=0
        for pre, exp, exp_t in zip(predictions, expected.data, expected.target):
          class_id = pre['class_ids'][0]
          probability = pre['probabilities'][class_id]
          print('Prediction is ', class_id, '(',probability,') and expected ', exp_t)
          if str(class_id) == str(exp_t):
             count+=1
             fout.writelines(','.join([str(x) for x in exp])+','+str(exp_t)+'\n')
    print('lenght of train set: ', len(expected.target))
    print('lenth of first ANN prediction: ', count)

def DNN2():
    process2csv('out_pre.txt', 'data_prediction_dnn1.csv')
    preprocess('data_prediction_dnn1.csv','dnn2')

    train_set = base.load_csv_with_header(filename='data_train_dnn2.csv',
                                     features_dtype=np.float32,
                                 target_dtype=np.int)
    predict_set = base.load_csv_with_header(filename='data_test_dnn2.csv',
                                     features_dtype=np.float32,
                                 target_dtype=np.int)
    classifier = model('/tmp/churn2')
    classifier.train(input_fn=input_fn(train_set),steps=1000000)

    accuracy_score = classifier.evaluate(input_fn=input_fn(predict_set),
                                     steps=10)["accuracy"]
    print('\nAccuracy: {0:f}'.format(accuracy_score))
    test_set = base.load_csv_with_header(filename='test_dnn2.csv',
                                             features_dtype=np.float32,
                                             target_dtype=np.int)
    predictions = classifier.predict(input_fn=input_fn(test_set))
    for pre, exp, exp_t in zip(predictions, expected.data, expected.target):
          class_id = pre['class_ids'][0]
          probability = pre['probabilities'][class_id]
          print('Prediction is ', class_id, '(',probability,') and expected ', exp_t)

def accuracy(predictions, labels):
    correct = np.sum(predictions==labels)
    total = predictions.shape[0]
    acc = 100*float(correct)/float(total)
    return acc

def DecissionTree():
    process2csv('out_pre.txt', 'data_tree.csv')
    preprocess('data_tree.csv','tree')
    train = csv.reader(open('data_train_tree.csv'), delimiter=",")
    train_set = np.array(list(train)).astype("float")
    test = csv.reader(open('data_test_tree.csv'), delimiter=",")
    test_set = np.array(list(test)).astype("float")
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_set[:,:-1], train_set[:,-1])

    with open("churn_classifier.txt", "w") as f:
       f = tree.export_graphviz(clf, out_file=f, feature_names=['Account Length','International Plan','VMail Plan','VMail Message','Day Mins','Day Calls','Day Charge','Eve Mins','Eve Calls','Eve Charge','Night Mins','Night Calls','Night Charge','Intl Mins','Intl Calls','Intl Charge','CustServ Calls'],
                         class_names='Churn',  
                         filled=True, rounded=True,  
                         special_characters=True)

    acc = accuracy(clf.predict(test_set[:,:-1]), test_set[:,-1])
    print('Accuracy: ', acc)

    pre = csv.reader(open('data_predict_tree.csv'), delimiter=",")
    predict = np.array(list(pre)).astype("float")
    prediction = clf.predict(predict[:,:-1])
    acc2 = accuracy(prediction, predict[:,-1])
    print('Prediction: ', acc2)

    prediction = [[i] for i in prediction]
    a =np.concatenate([predict, prediction], axis=1)
    np.savetxt("output_prediction.csv", a, delimiter=",")

	
if __name__=='__main__':
  #standard_data('metadata.csv')
  #DNN1()
  #DNN2()
  DecissionTree()
