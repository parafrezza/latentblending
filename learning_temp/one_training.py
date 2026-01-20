import torch
import torch.nn as nn


# Define a multi.layer neural network
class MultiLayerNN(nn.Module):
    def __init__(self):
        super(MultiLayerNN, self).__init__()
        self.layer1 = nn.Linear(3, 4)   # input layer to hidden layer
        self.layer2 = nn.Linear(4, 1)   # hidden layer to output layer
        self.relu = nn.ReLU()           # activation function

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x




#  datset
inputs  = torch.tensor( [
                         [1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]
                         ])

targets = torch.tensor([[10.0], [2.0], [30.0]])

# simple neural network

network   = MultiLayerNN()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(network.parameters(), lr=0.01)

# training loop
for epoch in range(100):           # 100 epochs
    optimizer.zero_grad()           # reset gradients
    outputs = network(inputs)      # forward pass
    loss = criterion(outputs, targets) # compute loss
    loss.backward()         # backpropagation
    optimizer.step()       # update weights
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# quick evaluation of the trained network (verbose)
with torch.no_grad():
    test_inputs = torch.tensor([
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0],
    ])
    print("\nInference walkthrough")
    print("Input:\n", test_inputs)
    print("Input shape:", tuple(test_inputs.shape))
    layer1_out = network.layer1(test_inputs)
    print("After layer1 (Linear 3->4):\n", layer1_out)
    print("Layer1 shape:", tuple(layer1_out.shape))
    relu_out = network.relu(layer1_out)
    print("After ReLU:\n", relu_out)
    print("ReLU shape:", tuple(relu_out.shape))
    layer2_out = network.layer2(relu_out)
    print("After layer2 (Linear 4->1):\n", layer2_out)
    print("Layer2 shape:", tuple(layer2_out.shape))
    print("Predictions:\n", layer2_out)
    print("Summary (input -> prediction):")
    for inp, pred in zip(test_inputs, layer2_out):
        print(f"  {inp.tolist()} -> {pred.item():.4f}")
