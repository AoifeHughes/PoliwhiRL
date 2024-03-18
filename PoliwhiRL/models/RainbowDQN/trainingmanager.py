import torch

class TrainingManager:
    def __init__(self, policy_net, target_net, replay_buffer, optimizer, config):
        self.policy_net = policy_net
        self.target_net = target_net
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.config = config
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def store_experience(self, state, action, reward, next_state, done):
        # Append experiences to the containers
        self.states.append(state)
        self.actions.append(torch.tensor([action], device=self.config["device"], dtype=torch.long))
        self.rewards.append(torch.tensor([reward], device=self.config["device"], dtype=torch.float))
        self.next_states.append(next_state)
        self.dones.append(torch.tensor([done], device=self.config["device"], dtype=torch.bool))

        # Check if we have enough experiences for a batch
        if len(self.states) >= self.config['batch_size']:
            self.compute_and_store_td_errors()

    def compute_and_store_td_errors(self):
        # Convert lists to tensors
        states = torch.stack(self.states).to(self.config["device"])
        actions = torch.stack(self.actions).to(self.config["device"])
        rewards = torch.stack(self.rewards).to(self.config["device"]).squeeze(-1)
        next_states = torch.stack(self.next_states).to(self.config["device"]).squeeze(-1)
        dones = torch.stack(self.dones).to(self.config["device"]).squeeze(-1)

        # Compute TD errors for the batch
        td_errors = self.compute_td_error(
            states,
            actions,
            rewards,
            next_states,
            dones,
            self.config["gamma"]
        )

        # Store experiences and TD errors in the replay buffer
        for state, action, reward, next_state, done, td_error in zip(self.states, self.actions, self.rewards, self.next_states, self.dones, td_errors):
            self.replay_buffer.add(
                state, action, reward, next_state, done, td_error
            )

        # Clear the containers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()

    def compute_td_error(self, states, actions, rewards, next_states, dones, gamma=0.99):
        current_q_values = self.policy_net(states).gather(1, actions).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            # Properly use the boolean mask to zero out values for terminal states
            next_state_values = next_state_values * (~dones)
            expected_q_values = rewards + gamma * next_state_values
        td_errors = (expected_q_values - current_q_values).abs()
        return td_errors  # Return as a tensor



    def optimize_model(self, beta):
        if len(self.replay_buffer) < self.config["batch_size"]:
            return None
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.config["batch_size"], beta)

        states = torch.stack(states).to(self.config["device"])
        actions = torch.stack(actions).to(self.config["device"])
        rewards = torch.stack(rewards).to(self.config["device"])
        next_states = torch.stack(next_states).to(self.config["device"])
        dones = torch.stack(dones).to(self.config["device"])
        weights = torch.FloatTensor(weights).unsqueeze(-1).to(self.config["device"])

        current_q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.policy_net(next_states).detach()
        _, best_actions = next_q_values.max(1, keepdim=True)
        next_q_values_target = self.target_net(next_states).detach().gather(1, best_actions)

        expected_q_values = rewards + (self.config['gamma'] * next_q_values_target * (~dones)).float()
        loss = (current_q_values - expected_q_values).pow(2) * weights
        prios = loss + 1e-5  # Avoid zero priority
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

        return loss.item()  # Optional: return the loss value for monitoring

    def reset_nets(self):
        self.policy_net.reset_noise_and_hidden(self.config["batch_size"])
        self.target_net.reset_noise_and_hidden(self.config["batch_size"])