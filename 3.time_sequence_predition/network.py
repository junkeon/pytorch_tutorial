import torch
import torch.nn as nn
import torch.nn.functional as F

class Sequence(nn.Module):
    def __init__(self, device):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)
        self.device = device
        
    def forward(self, inputs, future=0):
        outputs = []
        
        h_t1 = torch.zeros(inputs.size(0), 51).to(self.device)
        c_t1 = torch.zeros(inputs.size(0), 51).to(self.device)
        
        h_t2 = torch.zeros(inputs.size(0), 51).to(self.device)
        c_t2 = torch.zeros(inputs.size(0), 51).to(self.device)
        
        for i, input_t in enumerate(inputs.chunk(inputs.size(1), dim=1)):
            h_t1, c_t1 = self.lstm1(input_t, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
            
        for i in range(future):
            h_t1, c_t1 = self.lstm1(output, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)
            
        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs