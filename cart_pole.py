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
import torch.nn.functional as F



class Memory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = deque([], maxlen=self.capacity)
		self.counter = 0
		self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
	def push(self, *args):
		self.memory.append(self.Transition(*args))

	def sample(self, bach_size):
		return random.sample(self.memory, batch_size)
	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

	def __init__(self, n_obs, n_act, lr):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_obs,128)
		self.layer2 = nn.Linear(128,128)
		self.layer3 = nn.Linear(128, n_act)
		
		#self.criterion = nn.SmoothL1Loss()
		self.criterion = nn.MSELoss()		
	
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		
		self.to('cpu')
		
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		return self.layer3(x)


def select_action(state, t, eps_end, eps_start, eps_decay):
	
	sample = random.random()
	eps_thres = eps_end + (eps_start - eps_end) * math.exp(-1. *t/eps_decay)
		
		
	if sample > eps_thres:
		with torch.no_grad():
			x = policy_net.forward(state).max(1)[1].view(1,1)
			#print(x)
			return x
		
	else:
		x = torch.tensor([[env.action_space.sample()]], device = device, dtype=torch.long)
		#print(x)
		return x


def plot_durations(eps_dur, show_result=False):
	plt.figure(1)
	dur_t = torch.tensor(eps_dur, dtype=torch.float)
	if show_result:
		plt.title('Result')
		plt.savefig("figs/latest.png")
	else:
		plt.clf()
		plt.title('Training...')
		plt.xlabel('Episode')
		plt.ylabel('Duration')
		plt.plot(dur_t.numpy())
	
	if len(dur_t) >= 50:
		means = dur_t.unfold(0,50,1).mean(1).view(-1)
		means = torch.cat((torch.zeros(49), means))
		plt.plot(means.numpy())
		
	plt.pause(0.001)

def optimize_model(m, batch_size, policy_net, target_net, gamma):

	if len(m)<batch_size:
		return
	transitions = m.sample(batch_size) # make output the zipped tuple
	
	batch = m.Transition(*zip(*transitions)) # move into Memory class
	
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	#print(non_final_mask)
	state_batch = torch.cat(batch.state)#.squeeze(1)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)
	#print(action_batch)
	#print(state_batch)
	state_action_values = policy_net.forward(state_batch).gather(1, action_batch) #This is causing an error becuase dimensions don't match for this operation
	#print(state_action_values.shape)
	next_state_values = torch.zeros(batch_size, device=device)
	#print(state_action_values.shape)
	with torch.no_grad():
		next_state_values[non_final_mask] = target_net.forward(non_final_next_states).max(1).values
	#print(next_state_values)
	expected_state_action_values = (next_state_values * gamma) + reward_batch
	#print(expected_state_action_values)
	
	#print(non_final_mask.shape)
	optimizer = policy_net.optimizer
	
	loss = policy_net.criterion(expected_state_action_values.unsqueeze(1), state_action_values).to('cpu')
	#print(loss)
	optimizer.zero_grad()
	
	loss.backward()
	
	torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
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
	lr = 1e-2
	device = "cpu"
	num_episodes = 1000
	num_steps = 300

	# Initialize gym environment
	env = gym.make('CartPole-v1', render_mode="rgb_array")
	
	memory = Memory(10000)

	# Get size of action and observation spaces from the gym
	n_act = env.action_space.n
	state, info = env.reset()
	n_obs = len(state)

	# Initialize DQN Agent
	policy_net = DQN(n_obs, n_act, lr).to(device)
	target_net = DQN(n_obs, n_act, lr).to(device)
	target_net.load_state_dict(policy_net.state_dict())

	s = 0 
	eps_dur = []

	is_ipython = 'inline' in matplotlib.get_backend()
	if is_ipython:
		from IPython import display
	plt.ion()

	reward_hist = []
	done = False
	duration=[]

	for i_episode in range(num_episodes):
		
		print(i_episode)
		state, info = env.reset()
		#  Initial state
		total_reward = 0
		state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
		#print(state)
		for t in count():
			#if i_episode >= 250:
				#env.render()
			
			action = select_action(state, t, eps_end, eps_start, eps_decay)
			#print(action.shape)
			observation, reward,terminated,truncated,info = env.step(action.item())
			total_reward += reward
			
			if terminated or truncated:
				next_state = None
				done = True
				
			else:
				next_state = torch.tensor(observation, dtype=torch.float32, device = device).unsqueeze(0)
			
			reward = torch.tensor([reward], device="cpu")
			state = torch.tensor(state, dtype=torch.float32, device = device)#.unsqueeze(0)
			#print(state.shape)
			memory.push(state, action, next_state, reward)

			state = next_state

			optimize_model(memory, batch_size, policy_net, target_net, gamma)

			target_net_state_dict = target_net.state_dict()
			policy_net_state_dict = policy_net.state_dict()

			for key in policy_net_state_dict:
				target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
			target_net.load_state_dict(target_net_state_dict)

			if done:
				duration.append(t+1)
				plot_durations(duration)
				done = False
				break

		reward_hist.append(total_reward)

	print('Complete')
	#print(Average(reward_hist))
	plot_durations(duration, show_result=True)
	#plt.ioff()
	plt.show()


