# DQN

Deep Q-learning for discrete action spaces, one of the first mainstream deep
RL algorithms.

<p>
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="Aqua divider">
</p>

<p align="center">
  <img src="visual.gif" alt="DQN agent solving CartPole" width="384">
</p>

## Run summary

| Item | Result |
| --- | --- |
| Training steps | 1,000,000 |
| Training time | 7 seconds |
| Speed | 43,000 SPS |
| Solved | ✅ |

## Environment

- Package: Gymnax
- Environment: CartPole-v1
- Action space: Discrete

<p>
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="Aqua divider">
</p>

## Core algorithm

The target value is calculated with a separate target network:

```text
y = r + gamma * (1 - done) * max Q_target(s', a')
```

The online network is trained with the squared temporal difference error:

```text
L = mean((y - Q(s, a))^2)
```

The target network is copied from the online network at a fixed gradient
update interval.

<p>
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" alt="Aqua divider">
</p>

## Run

```bash
uv run python -m dqn.main --rng 0
```
