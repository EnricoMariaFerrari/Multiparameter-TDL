import torch
import torch.nn.functional as F

def signed_measures_formatter(data):
    """
    data: tuple of n elements
          each element = ((points, weights),)
            - points: torch.Tensor shape (m_i, k)
            - weights: torch.Tensor shape (m_i,)
    
    Output: torch.Tensor shape (n, m_max, k+1)
            where m_max = max_i m_i
            missing entries are padded with zeros
    """
    blocks = []
    # Find the maximum number of points across all measures
    m_max = max(points.shape[0] for ((points, weights),) in data)

    for ((points, weights),) in data:   # note the double nesting
        points = points.to(torch.float64)
        weights = weights.to(torch.float64).unsqueeze(1)
        pw = torch.cat([points, weights], dim=1)  # (m_i, k+1)

        m_i, dim = pw.shape
        if m_i < m_max:
            # Pad along the row dimension so that all tensors have size (m_max, k+1)
            pad_size = (0, 0, 0, m_max - m_i)  # (dim-wise)
            pw = torch.nn.functional.pad(pw, pad_size, value=0.0)

        blocks.append(pw)

    # Stack all into a single tensor: shape (n, m_max, k+1)
    return torch.stack(blocks, dim=0)