import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

#Define the number of features
features = 2

#Create the data
x = torch.tensor([[2, 4], [3, 9], [4, 16], [6, 36], [7, 49]], dtype=torch.float32)
y = torch.tensor([[70], [110], [165], [390], [550]], dtype=torch.float32)

#Initialize weights and Bias manually
w = torch.zeros((features, 1), requires_grad = True)
b = torch.zeros(1, requires_grad = True)
#Define the optimizer
optimizer = optim.SGD([w,b], lr=0.001)

#Training loop
for i in range(100000):
	#Forward pass: Matrix miltiplication and bias addition
	y_pred = x @ w + b #Matrix multiplication with bias addition
	#compute the loss
	loss = ((y_pred-y)**2).mean()
	#backward pass and optimization
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	#Print the paremeters every 10000 steps
	if i % 10000 == 0:
		with torch.no_grad():
			print(f'Iteration {i}, w: {w.detach().numpy()}, b: {b.item()}, loss: {loss.item()}')
			