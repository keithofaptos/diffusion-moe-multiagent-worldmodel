import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, num_experts=8, top_k=2, hidden_dim=512):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gates = torch.softmax(self.gate(x), dim=-1)
        topk_gates, topk_indices = torch.topk(gates, self.top_k, dim=-1)
        topk_gates = topk_gates / topk_gates.sum(dim=-1, keepdim=True)
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                output[mask] += topk_gates[mask][:, :(topk_indices[mask] == i).sum(dim=-1)].unsqueeze(-1) * expert(x[mask])
        return output
