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
    run_eval(model, env, config)

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

    run_eval(model, env, config)        
    torch.save(model.state_dict(), "ppo_pokemon_model.pth")




def run_eval(model, env, config):
    num_eval_episodes = config.get('num_eval_episodes', 10)
    sequence_length = config['sequence_length']
    device = config['device']

    model.eval()  # Set the model to evaluation mode
    
    total_rewards = []
    for episode in range(num_eval_episodes):
        state = env.reset()
        episode_rewards = 0
        done = False
        states_seq = []

        with torch.no_grad():  # No need to track gradients during evaluation
            while not done:
                state_tensor = image_to_tensor(state, device)
                states_seq.append(state_tensor)
                if len(states_seq) < sequence_length:
                    continue  # Wait until we have enough states for a full sequence

                state_sequence_tensor = torch.stack(states_seq[-sequence_length:]).unsqueeze(0)

                action_probs, _ = model(state_sequence_tensor)
                action = torch.distributions.Categorical(action_probs[0]).sample().item()

                next_state, reward, done = env.step(action)
                env.record(0, "ppo_eval", False, 0)
                episode_rewards += reward
                state = next_state

                if len(states_seq) == sequence_length:
                    states_seq.pop(0)  # Keep the sequence buffer at fixed size

        total_rewards.append(episode_rewards)    
    avg_reward = sum(total_rewards) / num_eval_episodes
    model.train()  # Set the model back to training mode
    return avg_reward
