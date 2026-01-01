import torch

def compute_drift(phi_t, phi_prev):
    return torch.norm(phi_t - phi_prev, p=2)

def h7_gate(drift, threshold=0.1):
    return "approve" if drift < threshold else "reject"
