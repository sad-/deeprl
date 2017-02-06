import pickle
import tensorflow as tf
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_dump', type=str)
    args = parser.parse_args()
    training_data = pickle.load(open(args.expert_policy_dump, 'rb'))
    clone(training_data)

def clone(training_data):
    print "loading data"
    observations = training_data['observations']
    observations = np.hstack((observations, np.ones((observations.shape[0],1)))) #append ones vector
    actions = training_data['actions']
    actions = actions.reshape((actions.shape[0], actions.shape[-1]))
    N = observations.shape[0]
    m = observations.shape[-1]
    l = 10
    k = actions.shape[-1]
    print actions.shape
    print observations.shape
    test_indices = np.random.choice(N, N//5, replace=False)
    train_indices = np.array([i for i in xrange(N) if i not in test_indices])
    observations = {'train':observations[train_indices, :], 'test':observations[test_indices,:]}
    actions = {'train':actions[train_indices,:], 'test':actions[test_indices,:]}
    print "done loading data"

    print "constructing graph"
    x = tf.placeholder("float", shape = [None, m])
    y_ = tf.placeholder("float", shape = [None, k])
    w_1 = tf.Variable(tf.random_normal((m,l), stddev=0.1), name="w1")
    w_2 = tf.Variable(tf.random_normal((l, k), stddev=0.1), name="w2")
    h = tf.nn.sigmoid(tf.matmul(x, w_1))
    y = tf.matmul(h, w_2)
    loss = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


    print "creating session"
    session = tf.Session()

    print "initializing variables"
    session.run(tf.initialize_all_variables())

    print "training the model"
    with session.as_default():
        for i in range(50000):
            if i%1000==0:
                training_loss = loss.eval(feed_dict={x:observations['train'], y_:actions['train']})
                print "training loss after %d epochs: %g"%(i, training_loss)
            train_step.run(feed_dict={x:observations['train'], y_:actions['train']})

    print "test loss %g"%loss.eval(session=session, feed_dict={x:observations['test'], y_:actions['test']})
    saver = tf.train.Saver()
    save_path = saver.save(session, "/tmp/model.ckpt")
    print "Model saved in file: %s"%save_path

if __name__ == '__main__':
    main()