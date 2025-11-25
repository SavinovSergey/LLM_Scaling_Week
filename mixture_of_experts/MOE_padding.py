import torch
import time
from typing import Tuple, List, Dict


def moe_padding(
    x: torch.Tensor,
    top_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    topk: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hidden_size = x.shape
    
    # Compute padded tokens per expert (round up to nearest multiple of 128)
    # Убедимся, что тип данных совпадает с naive версией
    padded_tokens_per_expert = ((tokens_per_expert + 127) // 128) * 128
    padded_tokens_per_expert = padded_tokens_per_expert.to(torch.int32)
    
    total_padded = padded_tokens_per_expert.sum().item()
    
    # Initialize output tensor with zeros
    output = torch.zeros(total_padded, hidden_size, dtype=x.dtype, device=x.device)
    
    # Compute expert offsets in the output tensor
    expert_offsets = torch.cumsum(padded_tokens_per_expert, dim=0) - padded_tokens_per_expert
    
    # Flatten the top_experts and create corresponding token indices
    flat_top_experts = top_experts.flatten()  # (num_tokens * topk)
    flat_token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, topk).flatten()
    
    # Sort by expert to group tokens for the same expert together
    sorted_expert, sort_indices = torch.sort(flat_top_experts)
    sorted_token_indices = flat_token_indices[sort_indices]
    
    # Compute positions in output tensor using unique_consecutive
    unique_ids, inverse_ids, counts = torch.unique_consecutive(
        sorted_expert, return_inverse=True, return_counts=True
    )
    
    # Calculate output indices
    global_indices = torch.arange(len(sorted_expert), device=x.device)
    expert_start_indices = torch.cat([torch.tensor([0], device=x.device), counts.cumsum(0)[:-1]])
    start_indices = expert_start_indices[inverse_ids]
    local_indices = global_indices - start_indices
    output_indices = expert_offsets[sorted_expert] + local_indices
    
    # Write tokens to output positions
    output[output_indices] = x[sorted_token_indices]
    
    return output, padded_tokens_per_expert


def torch_basic(
    x: torch.Tensor, 
    top_experts: torch.Tensor, 
    tokens_per_expert: torch.Tensor, 
    topk: int, num_experts: int
)-> tuple[torch.Tensor, torch.Tensor]:
    block_size = 128
    device = x.device
    num_tokens, hidden_dim = x.shape

    expert_ids_flat = top_experts.view(-1)

    padded_tokens_per_expert = (
        ((tokens_per_expert + block_size - 1) // block_size) * block_size
    ).to(torch.int32)
    padded_offsets = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        padded_tokens_per_expert.cumsum(dim=0)
    ])
    expert_ids_cpu = expert_ids_flat.cpu().tolist()
    padded_offsets_cpu = padded_offsets.cpu().tolist()

    max_padded_tokens = padded_offsets_cpu[-1]
    padded_tokens = torch.zeros(
        (max_padded_tokens, hidden_dim),
        dtype=x.dtype,
        device=device,
    )

    assignment_groups = [[] for _ in range(num_experts)]
    num_assignments = topk * num_tokens
    for i in range(num_assignments):
        expert_id = expert_ids_cpu[i]
        assignment_groups[expert_id].append(i)

    for e in range(num_experts):
        local_idx = 0
        offset = padded_offsets[e]

        for local_idx, i in enumerate(assignment_groups[e]):
            original_token_idx = i // topk
            token_data = x[original_token_idx]
            target_row = offset + local_idx
            padded_tokens[target_row, :] = token_data

    return padded_tokens, padded_tokens_per_expert