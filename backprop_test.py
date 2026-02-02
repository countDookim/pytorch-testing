import torch
import matplotlib.pyplot as plt

x = torch.tensor([[7.01], [3.02], [4.99], [8.0]], dtype=torch.float32)
y = torch.tensor([[14.01], [6.0], [10.01], [16.04]], dtype=torch.float32)

w = torch.tensor([1.0], dtype=torch.float32, requires_grad = True)

losses = list()
weights = list()

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
	losses.append(loss.item())
	weights.append(w.item())
	print(f'Epoch {epoch}: loss = {loss.item()} w = {w.item()}')

plt.figure(figsize = (12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(weights, label="weight (w)", color = "orange")
plt.xlabel("Epoch")
plt.ylabel("W")
plt.title("Weight (w) over Epoch")
plt.legend()

plt.show()
