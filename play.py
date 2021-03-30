import pybullet_envs  # noqa
import gym
import numpy as np
import os
from gerel.util.datastore import DataStore
from gerel.model.model import Model
from gym.wrappers import Monitor
import click


#  https://www.etedal.net/2020/04/pybullet-panda_2.html

ENV_NAME = 'AntBulletEnv-v0'
STEPS = 1000


def play(genome, record=False, steps=1000):
    done = False
    model = Model(genome)
    env = gym.make(ENV_NAME)
    if record:
        env = Monitor(env, './video', force=True)
    env.render()
    state = env.reset()
    action_map = lambda x: np.tanh(np.array(x))  # noqa
    rewards = 0
    i = 0
    while not done and i < STEPS:
        i += 1
        action = model(state)
        action = action_map(action)
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        state = next_state
        env.render()
    return rewards


@click.command()
@click.option('--record', '-r', is_flag=True,
              help='Record roleout')
@click.option('--steps', '-s', default=1000, type=int,
              help='Max number of steps per episode')
@click.option('--generation', '-g', default=None, type=int,
              help='Generation to play')
def cli(record, steps, generation):
    if not generation:
        generation = max([int(i) for i in os.listdir('./ant_RES_data')])

    ds = DataStore(name='ant_RES_data')
    data = ds.load(generation)
    rewards = play(data['best_genome'], record, steps)
    print(f'generation: {generation}, rewards: {rewards}')


if __name__ == '__main__':
    cli()
