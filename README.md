<h1 align="center">Fast Reinforcement Learning</h1>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="Aqua divider">
</p>

<p align="center">
  <strong>JAX</strong><br>
  A few reinforcement learning algorithms written in JAX, following a functional, stateless style.
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="Aqua divider">
</p>

## Implementations

| Algorithm | Environment | Environment package | Action space |
| --- | --- | --- | --- |
| [DQN](dqn/README.md) | CartPole-v1 | Gymnax | Discrete |
| [DDPG](ddpg/README.md) | HalfCheetah | Brax with MJX | Continuous |
| [PPO](ppo/README.md) | Humanoid | Brax with MJX | Continuous |

<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="Aqua divider">
</p>

## Usage

```bash
uv sync

uv run python -m dqn.main --rng 0
uv run python -m ddpg.main --rng 0
uv run python -m ppo.main --rng 0
```
