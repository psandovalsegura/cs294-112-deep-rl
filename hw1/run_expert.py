#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
from tensorflow import keras
import numpy as np
import tf_util
import gym
import load_policy

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--expert_already_trained', action='store_true')
    args = parser.parse_args()

    with tf.Session() as sess:
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        # Follow expert policy to generate rollouts
        generate_expert_rollout(args, max_steps, env)

        # Get expert data
        observations, actions = get_expert_data(args)

        # Create model
        model = keras.Sequential([
            keras.layers.Dense(5),
            keras.layers.Dense(3)
        ])

        actions_pred = model(observations)
        loss = tf.losses.mean_squared_error(labels=actions, predictions=actions_pred)
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

        # Use supervised training to learn the expert policy
        init = tf.global_variables_initializer()
        sess.run(init)
        print('Training...')
        for i in range(50000):
            _, loss_value = sess.run((train, loss))
            if i % 500 == 0:
                print(loss_value)
        print(loss_value)

        # Generate rollouts for the trained classifier
        evaluate(args, max_steps, env, sess, model)

def generate_expert_rollout(args, max_steps, env):

    if not args.expert_already_trained:
        print('loading and building expert policy')
        policy_fn = load_policy.load_policy(args.expert_policy_file)
        print('loaded and built')

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('Expert rollouts' + '\n--------------------')
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)    

def get_expert_data(args):
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
            expert_data = pickle.load(f)

    observations = expert_data['observations']
    actions = expert_data['actions']

    print(f'Observations:  with shape {observations.shape} type {type(observations)}')
    print(f'Actions:  with shape {actions.shape} type {type(actions)}')

    observations = tf.constant(observations)
    actions = np.squeeze(actions, axis=1)
    actions = tf.constant(actions)

    observations = tf.cast(observations, tf.float32)
    actions = tf.cast(actions, tf.float32)

    return observations, actions


def evaluate(args, max_steps, env, sess, model):
    returns = []

    for i in range(args.num_rollouts):
            print('eval iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                observation = tf.cast(tf.constant(np.expand_dims(obs[None,:], axis=0)), tf.float32)
                action = sess.run(model(observation))
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

    print('Evaluated clone' + '\n--------------------')
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()
