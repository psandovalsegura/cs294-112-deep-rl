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
from tqdm import tqdm 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--use_dagger', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    with tf.Session() as sess:
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        # Follow expert policy to generate rollouts
        expert_data = None
        if args.use_dagger:
            expert_data = generate_expert_rollout(args, max_steps, env)
        else:
            generate_expert_rollout(args, max_steps, env)

        # Get expert data
        observations, actions = get_expert_data(args, expert_data)

        # Create model
        model = keras.Sequential([
            keras.layers.Dense(100),
            keras.layers.Dense(100),
            keras.layers.Dense(3)
        ])

        actions_pred = model(observations)
        loss = tf.losses.mean_squared_error(labels=actions, predictions=actions_pred)
        optimizer = tf.train.GradientDescentOptimizer(0.01)
        train = optimizer.minimize(loss)

        # Use supervised training to learn the expert policy
        init = tf.global_variables_initializer()
        sess.run(init)

        # Evaluate initial model
        # evaluate(args, max_steps, env, sess, model, name='random')
        
        training_iter = 2500
        print(f'Training with {observations.get_shape()} observations...')
        for i in tqdm(range(training_iter), desc='training'):
            _, loss_value = sess.run((train, loss))
            if i % 100 == 0 and args.verbose:
                print(loss_value)
        print(loss_value)

        if args.use_dagger:
            dagger_iterations = 5
            agg_observations = observations
            agg_actions = actions
            for i in tqdm(range(dagger_iterations), desc='dagger iteration'):
                # Generate new rollouts and append to dataset
                expert_data = generate_expert_rollout(args, max_steps, env)
                new_observations, new_actions = get_expert_data(args, expert_data)
                agg_observations = tf.concat([agg_observations, new_observations], 0)
                agg_actions = tf.concat([agg_actions, new_actions], 0)
                agg_size = agg_observations.get_shape()

                actions_pred = model(agg_observations)
                loss = tf.losses.mean_squared_error(labels=agg_actions, predictions=actions_pred)
                train = optimizer.minimize(loss)

                # Use supervised training to learn the expert policy
                print(f'Training dagger iteration {i} of {agg_size} observations...')
                for i in tqdm(range(training_iter), desc='dagger training'):
                    _, loss_value = sess.run((train, loss))
                    if i % 100 == 0 and args.verbose:
                        print(loss_value)
                print(loss_value)

        # Generate rollouts for the trained classifier
        evaluate(args, max_steps, env, sess, model, name='trained')

def generate_expert_rollout(args, max_steps, env):

    if args.verbose: 
        print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)

    returns = []
    observations = []
    actions = []
    for i in tqdm(range(args.num_rollouts), desc='expert rollouts'):
        if args.verbose: print('iter', i)
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
            if steps % 100 == 0 and args.verbose: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('Generated expert rollouts' + '\n--------------------')
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    if not args.use_dagger:
        with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)    

    return expert_data

def get_expert_data(args, expert_data=None):
    if not args.use_dagger:
        with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
            expert_data = pickle.load(f)

    observations = expert_data['observations']
    actions = expert_data['actions']

    if args.verbose:
        print(f'Observations:  with shape {observations.shape} type {type(observations)}')
        print(f'Actions:  with shape {actions.shape} type {type(actions)}')

    observations = tf.constant(observations)
    actions = np.squeeze(actions, axis=1)
    actions = tf.constant(actions)

    observations = tf.cast(observations, tf.float32)
    actions = tf.cast(actions, tf.float32)

    return observations, actions


def evaluate(args, max_steps, env, sess, model, name='clone'):
    returns = []

    for i in tqdm(range(args.num_rollouts), desc='eval'):
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

    print('Evaluated '+ name + '\n--------------------')
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

if __name__ == '__main__':
    main()
