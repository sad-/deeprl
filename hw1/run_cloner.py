import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import cloner
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('expert_policy_dump', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--daggr', action='store_true')
    parser.add_argument('--pretrained', action='store_true')

    args = parser.parse_args()

    #run only if daggr - change later
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    #prep for cloning
    print "loading expert data"
    training_data = pickle.load(open(args.expert_policy_dump, 'rb'))
    old_obs = training_data['observations']
    training_data['observations'] = np.hstack((training_data['observations'], np.ones((training_data['observations'].shape[0],1))))
    training_data['actions'] = training_data['actions'].reshape((training_data['actions'].shape[0], training_data['actions'].shape[-1]))
    m = training_data['observations'].shape[-1]
    l = 10
    k = training_data['actions'].shape[-1]

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
    saver = tf.train.Saver()

    def clone(training_data):
        print "creating training and test data"
        observations = training_data['observations']
        actions = training_data['actions']
        N = observations.shape[0]
        test_indices = np.random.choice(N, N//5, replace=False)
        train_indices = np.array([i for i in xrange(N) if i not in test_indices])
        observations = {'train':observations[train_indices, :], 'test':observations[test_indices,:]}
        actions = {'train':actions[train_indices,:], 'test':actions[test_indices,:]}

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
        return save_path

    with session:
        if args.pretrained:
            save_path = "/tmp/model.ckpt"
        else:
            save_path = clone(training_data)
        saver.restore(session, save_path)
        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        expert_actions = []

        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                observation = np.hstack((obs[None,:], np.ones((1,1))))
                action = y.eval(session=session, feed_dict={x:observation})
                expert_action = policy_fn(obs[None,:])
                #action1 = y.eval(feed_dict={x:training_data['observations'][0,:].reshape((1, 12))})
                #expert_action1 = policy_fn(old_obs[0,:].reshape(1, 11))
                observations.append(obs)
                actions.append(action)
                expert_actions.append(expert_action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
                if args.daggr and steps%200==0:
                    training_data['observations'] = np.concatenate(training_data['observations'], np.array(observations))
                    training_data['actions'] = np.concatenate(training_data['actions'], np.array(expert_actions))
                    save_path = clone(training_data)
                    observations = []
                    expert_actions = []
                returns.append(totalr)
        #print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

if __name__ == '__main__':
    main()