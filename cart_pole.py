import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F



class Memory(object):
	def __init__(self, capacity):
		self.memory = deque([], maxlen=capacity)
        self.counter = 0
	
	def push(self, args):
		self.memory.append(Transition(args))
	
	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)
	
	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

	def __init__(self, n_obs, n_act):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_obs,128)
		self.layer2 = nn.Linear(128,128)
		self.layer3 = nn.Linear(128, n_act)
        
        self.optimizer = optim.AdamW(self.paramters(), lr=lr, amsgrad=True)
        self.loss = nn.SmoothL1Loss()
        self.device = torch.device('cpu')
        self.to(self.device)
        
		
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)


def select_action(state):
	global s, env
	sample = random.random()
	eps_thres = eps_end + (eps_start - eps_end) * math.exp(-1*s/eps_decay)
	s+=1
	
	if sample > eps_thres:
		with torch.no_grad():
			return policy_net(state).max(1)[1].view(1,1)
		
	else:
		return torch.tensor([[env.action_space.sample()]], device = device, dtype=torch.long)


def plot_durations(show_result=False):
	plt.figure(1)
	dur_t = torch.tensor(eps_dur, dtype=torch.float)
	if show_result:
		plt.title('Result')
	else:
		plt.clf()
		plt.title('Training...')
	plt.xlabel('Episode')
	plt.ylabel('Duration')
	plt.plot(dur_t.numpy())
	
	if len(dur_t) >= 100:
		means = dur_t.unfold(0,100,1).mean(1).view(-1)
		means = torch.cat((torch.zeros(99), means))
		plt.plot(means.numpy())
		
	plt.pause(0.001)

def optimize_model(batch_size):
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    memory = Replay(10000)
	if len(memory)<batch_size:
		return
	transitions = memory.sample(batch_size)
	
	batch = Transition(zip(transitions))
	
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)
	
	state_action_values = policy_net(state_batch).gather(1, action_batch)
	
	next_state_values = torch.zeros(batch_size, device=device)
	with torch.no_grad():
		next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
	
	expected_state_action_values = (next_state_values * gamma) + reward_batch
	
	loss = DQN.loss(state_action_values, expected_state_action_values.unsqueeze(1)).to("cpu")
	
	optimizer.zero_grad()
	loss.backward()
	
	torch.nn.utils.clip_grad_value(policy_net.parameters(),100)
	optimizer.step()
	

	

if __name__ == "__main__":

    global Transition

    ###############
    #   Parameters
    ###############
    batch_size = 128
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    tau = 0.005
    lr = 1e-4
    device = "cpu"
    num_episodes = 500
    num_steps = 300

    # Initialize gym environment
    env = gym.make('CartPole-v1')

    # Get size of action and observation spaces from the gym
    n_act = env.action_space.n
    state, info = env.reset()
    n_obs = len(state)

    # Initialize DQN Agent
    policy_net = DQN(n_obs, n_act).to(device)
    target_net = DQN(n_obs, n_act).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    s = 0 
    eps_dur = []

    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    
    reward_hist = []
    
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) #  Initial state
        total_reward = 0
        for t in count(num_steps ):
            if i_episode >= 250:
                env.render()
            action = select_action(state)
            observation, reward,done,truncated = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            
            if done:
                next_state = None
                break
           
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
                
            memory.push(state,action,next_state,reward)
            
            state = next_state
            
            optimize_model(batch_size)
            
            target_net_state_dist = target_net.state_dist()
            policy_net_state_dict = policy_net.state_dict()
            
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*tau + targe_net_state_dict[key]*(1-tau)
            target_net.load_state_dict(target_net_state_dict)
            
            if done:
                episode_durations.append(t+1)
                plot_durations()
                break
         
         reward_hist.append(total_reward)

    print('Complete')
    plot_durations(show_result=True)
    plt.ioff()
    plt.show()
                
