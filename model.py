import torch
import torch.nn as nn
""" Style 1: Usually used to write short neural networks fucntion
class Simple_Neural_Networks(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features= 3*32*32 , out_features=256)
        self.act1 = nn.ReLU() # Activation function doesn't change the shape.
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act1(x)
        return x
"""
# Style 2: Normally used
class Simple_Neural_Networks(nn.Module):
    def __init__(self, num_classess = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=3*32*32, out_features=256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features= 512),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features= 512, out_features= num_classess),
            nn.ReLU()
        )
    """
        Style 3: use for simple neural networks only
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3*32*32, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features= 512),
            nn.ReLU(),
            nn.Linear(in_features= 512, out_features= num_classess),
            nn.ReLU()
        )
    """
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x

if __name__ == "__main__":       
    model = Simple_Neural_Networks() 
    input_data = torch.rand(8,3,32,32)
    result = model(input_data)
    print(result)