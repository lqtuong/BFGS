import pandas as pd
import tensorflow as tf
import argparse

DATA_TRAIN = 'iris_training.csv'
DATA_TEST = 'iris_test.csv'

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
LABELS = ['Setosa', 'Versicolor', 'Virginica']

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
parser.add_argument('--train_steps', default=1000, type=int, help='number of training steps')

def load_data(label='Species'):
    train = pd.read_csv('~/Downloads/tacotron/iris_training.csv', names=COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(label)

    test = pd.read_csv('~/Downloads/tacotron/iris_test.csv', names=COLUMN_NAMES, header=0)
    test_x, test_y =  test, test.pop(label)

    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset

def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None"

    dataset = dataset.batch(batch_size)

    return dataset

def DNN(features, labels, mode, params):
    # NEURAL NET
    net = tf.feature_column.input_layer(features, params['feature_columns']) # features of dataset
    for units in params['hidden_units']: # hidden layer
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu) # put features to hidden units of hidden layer
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)

    # PREDICTION
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # LOSS
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op'
                                    )
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # EVALUATION
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # TRAINING
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    args = parser.parse_args(argv[1:])

    (train_x, train_y), (test_x, test_y) = load_data()

    feature_columns = []
    for key in train_x.keys(): # convert column names to number
        feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.Estimator(
        model_fn=DNN,
        params={
            'feature_columns': feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        }
    )

    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps
    )

    eval_result = classifier.evaluate(
        input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size)
    )

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x =  {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x, labels=None, batch_size=args.batch_size)
    )

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(LABELS[class_id], 100*probability, expec))

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)