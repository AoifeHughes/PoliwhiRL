import torch
import torch.optim as optim
from tqdm import tqdm
from PoliwhiRL.environment.controller import Controller as Env
from .PPO import PPOModel
from .training_functions import compute_returns
from PoliwhiRL.utils.utils import image_to_tensor
import numpy as np


def run(config):
    env = Env(config)
    state = env.reset()

    input_dim = (1 if config["use_grayscale"] else 3, *env.screen_size())
    output_dim = len(env.action_space)
    model = PPOModel(input_dim, output_dim).to(config["device"])
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_episodes = 200
    gamma = 0.99

    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        episode_rewards = 0
        saved_log_probs = []
        saved_values = []
        rewards = []
        masks = []
        done = False
        states_seq = []

        while not done:
            state_tensor = image_to_tensor(state, config["device"])
            states_seq.append(state_tensor)
            if len(states_seq) < config['sequence_length']:
                continue  # Accumulate enough states for a sequence before prediction

            # Adjust dimensions to fit [batch, seq_length, c, h, w]
            state_sequence_tensor = torch.stack(states_seq[-config['sequence_length']:]).unsqueeze(0)

            action_probs, value_estimates = model(state_sequence_tensor)
            dist = torch.distributions.Categorical(action_probs[0])
            action = dist.sample()

            next_state, reward, done = env.step(action.item())

            episode_rewards += reward
            saved_log_probs.append(dist.log_prob(action).unsqueeze(0))
            saved_values.append(value_estimates)
            rewards.append(reward)
            masks.append(1 - done)
            state = next_state

        next_value = saved_values[-1].detach()  # Use the last value estimate for computing returns
        returns = compute_returns(next_value, rewards, masks, gamma)

        log_probs = torch.cat(saved_log_probs)
        returns = torch.cat(returns)
        values = torch.cat(saved_values)

        advantages = returns - values
        action_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()

        loss = action_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Total Reward: {episode_rewards}")

    run_eval(model, env, config)
    torch.save(model.state_dict(), "ppo_pokemon_model.pth")