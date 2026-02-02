import torch

x = torch.tensor([[7.01], [3.02], [4.99], [8.0]], dtype=torch.float32)
y = torch.tensor([[14.01], [6.0], [10.01], [16.04]], dtype=torch.float32)

w = torch.tensor([1.0], dtype=torch.float32, requires_grad = True)

for epoch in range(100):
	y_pred=w*x
	#compute mean squared error loss
	loss=((y_pred-y)**2).mean()
	#backward pass to compute gradients
	loss.backward()
	#update weight using gradient descent
	with torch.no_grad():
		w-=0.001 * w.grad #update weight
		w.grad.zero_() #clear grad for next step
	print(f'Epoch {epoch}: loss = {loss.item()} w = {w.item()}')
