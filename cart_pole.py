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

###############
#   Parameters
###############
batch_size = 128
gamma = 0.99
eps_start = 1.0
eps_end = 0.05
eps_decay = .9
tau = .005
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_episodes = 1000
capacity = 10000
model_save = True

# Initialize gym environment
#env = gym.make('CartPole-v1', render_mode="rgb_array")
env = gym.make('CartPole-v1')
# Get size of action and observation spaces from the gym
n_act = env.action_space.n
state, info = env.reset()
n_obs = len(state)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

s = 0 

class Memory(object):
	def __init__(self):
		self.capacity = capacity
		self.memory = deque([], maxlen=self.capacity)
		
	def push(self, *args):
		self.memory.append(Transition(*args))

	def sample(self, bach_size):
		return random.sample(self.memory, batch_size)
	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):

	def __init__(self, n_obs, n_act):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_obs,batch_size)
		self.layer2 = nn.Linear(batch_size,batch_size)
		self.layer3 = nn.Linear(batch_size, n_act)
		self.to(device)

		
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = self.layer3(x)
		#print(x)
		return x


def select_action(state, policy_net):
	global eps_thres
	
	sample = random.random()
	eps_thres *= eps_decay 
	
	if eps_thres < eps_end:
		eps_thres = eps_end
		
		
	if sample > eps_thres:
		with torch.no_grad():
			x = policy_net(state).max(1)[1].view(1,1)
			#print(x)
			return x
		
	else:
		x = torch.tensor([[env.action_space.sample()]], device = device, dtype=torch.long)
		#print(x.shape)
		return x


def plot_durations(eps_dur, show_result=False):
	plt.figure(1)
	dur_t = torch.tensor(eps_dur, dtype=torch.float)
	if show_result:
		plt.title('Result')
		plt.savefig("figs/latest.png")
		print("Figure Saved")
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
		
	#plt.pause(0.001)
	
def save_model(policy_net):
	size = str(batch_size)
	lr_s = str(lr)
	
	modelfile = "models/cartmodel_" + size + "_" + lr_s + ".pt"
	torch.save(policy_net.state_dict(), modelfile)

	print("Model Saved as:", modelfile)

def optimize_model(policy_net, target_net):
	global memory, optimizer
	m = memory
	if len(m)<batch_size:
		return
	transitions = m.sample(batch_size) # make output the zipped tuple
	
	batch = Transition(*zip(*transitions)) # move into Memory class
	
	
	non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=device, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
	
	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)
	
	
	state_action_values = policy_net(state_batch).gather(1, action_batch) #This is causing an error becuase dimensions don't match for this operation
	
	next_state_values = torch.zeros(batch_size, device=device)
	
	next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1).values
	
	expected_state_action_values = (next_state_values * gamma) + reward_batch
	
	criterion = nn.SmoothL1Loss()
	#criterion = nn.MSELoss()		
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
	
	optimizer.zero_grad()	
			
	loss.backward()
		
	torch.nn.utils.clip_grad_value_(policy_net.parameters(),100)
	optimizer.step()
	
	target_net_state_dict = target_net.state_dict()
	policy_net_state_dict = policy_net.state_dict()
	#print(target_net_state_dict)

	for key in policy_net_state_dict:
		target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
	target_net.load_state_dict(target_net_state_dict)
	
	
	

	

if __name__ == "__main__":

	global memory, optimizer, eps_thres
	
	# Initialize DQN Agent
	policy_net = DQN(n_obs, n_act).to(device)
	target_net = DQN(n_obs, n_act).to(device)
	target_net.load_state_dict(policy_net.state_dict())
	
	optimizer = optim.AdamW(policy_net.parameters(), lr=lr)
	
	memory = Memory()
	is_ipython = 'inline' in matplotlib.get_backend()
	if is_ipython:
		from IPython import display
	plt.ion()

	reward_hist = []
	
	duration=[]
	print("Training...")
	for i_episode in range(num_episodes):
		eps_thres = eps_start
		if i_episode % 100 == 0:
			print(i_episode)
		state, info = env.reset()
		#  Initial state
		total_reward = 0
		state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
		
		
		for t in count():
			
			action = select_action(state, policy_net)
			
			observation, reward,terminated,truncated,info = env.step(action.item())
			total_reward += reward
			done = terminated or truncated

			if terminated:
				next_state = None
								
			else:
				next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
			
			reward = torch.tensor([reward], device=device)
			memory.push(state, action, next_state, reward)

			state = next_state

			optimize_model(policy_net, target_net)



			if done:
				duration.append(t+1)
				plot_durations(duration)
				
				break

		reward_hist.append(total_reward)

	print('Training Complete')
	
	if model_save: save_model(policy_net)
	#print(Average(reward_hist))
	plot_durations(duration, show_result=True)
	#plt.ioff()
	plt.show()


