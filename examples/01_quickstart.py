"""One episode with random actions: the standard gymnasium loop.

    uv run python examples/01_quickstart.py
    uv run python examples/01_quickstart.py --task=small --episode=3

The same episode index always replays the same scenario, the one ``sbf evaluate --episodes=[n]`` scores.
"""
# Good luck!

import fire
import gymnasium as gym
import shockbench_flow_gym  # noqa: F401 - registers the ShockBench/* environments

from sbf_starter import env_id


def main(task: str = "tiny", episode: int = 0, seed: int = 0) -> None:
    """Play one dev episode with random actions and print its cost."""
    env = gym.make(env_id(task))
    obs, info = env.reset(options={"episode": episode})
    env.action_space.seed(seed)
    print(f"{env_id(task)}: {len(obs)} observation arrays; action parts {list(env.action_space.spaces)}")
    print(f"dev episode {info['episode']}: env.reset(options={{'episode': {info['episode']}}}) replays it")
    total, weeks, done = 0.0, 0, False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)  # reward = minus this week's cost in USD
        total, weeks, done = total + reward, weeks + 1, terminated or truncated
    print(f"{weeks} weeks, total cost {-total:,.0f} USD with random actions (lower is better)")


if __name__ == "__main__":
    fire.Fire(main)
