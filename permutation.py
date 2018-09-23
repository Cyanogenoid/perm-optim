import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_assignment(elements, assignment):
    """ Apply an assignment on a set of elements to re-order them.
    """
    # expand elements over the positions it can take
    elements = elements.unsqueeze(3)
    # expand assignment over the elements it works over
    dims_to_append = elements.dim() - assignment.dim()
    assignment = assignment.view(*assignment.size(), *([1] * dims_to_append))
    # weighted sum over positions
    x = assignment * elements
    x = x.sum(dim=2)
    return x


def calculate_assignment(cost_matrix, assignment=None, lr=1, temp=1, steps=1, size_2d=None):
    """ Compute a good assignment for the given cost matrix.
    """
    # initialise assignment if necessary
    if assignment is None:
        size = cost_matrix.size(0), 1, cost_matrix.size(2), cost_matrix.size(3)
        assignment = cost_matrix.new(*size).fill_(0)
    else:
        size = assignment.size()

    # we don't care about scale of cost matrix, so normalise
    cost_matrix = cost_matrix / cost_matrix.view(*cost_matrix.size()[:2], -1).norm(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-10)

    if not size_2d:
        normalise = sinkhorn
        compute_grad = assignment_grad
    else:
        normalise = sinkhorn_2d
        compute_grad = assignment_grad_2d
        assignment = assignment.view(*size[:-1], *size_2d)

    # gradient descent
    for _ in range(steps):
        assignment_normed = normalise(assignment, temp=temp, steps=4)
        grad = compute_grad(assignment_normed, cost_matrix)
        assignment = assignment - lr * grad

    assignment = normalise(assignment, temp=temp, steps=4)
    if size_2d:
        assignment = assignment.view(*assignment.size()[:-2], -1)
    return assignment


def assignment_grad_2d(assignment, cost_matrix):
    # assignment :: Tensor(n, 1, i, row, col)
    # assume cost matrix has two channels, one for row costs and one for col costs
    row_cost_matrix = cost_matrix[:, :1, ...].contiguous()
    col_cost_matrix = cost_matrix[:, 1:, ...].contiguous()

    total_grad = assignment.new(assignment.size()).fill_(0)
    for idx in range(assignment.size(-1)):
        row_assignment = assignment[..., idx, :]
        col_assignment = assignment[..., :, idx]
        # to assign to a row, need to compare between cols
        row_grad = assignment_grad(row_assignment, col_cost_matrix)
        # to assign to a col, need to compare between rows
        col_grad = assignment_grad(col_assignment, row_cost_matrix)
        total_grad[..., idx, :] = total_grad[..., idx, :] + row_grad
        total_grad[..., :, idx] = total_grad[..., :, idx] + col_grad
    return total_grad


def assignment_grad(assignment, cost_matrix, local=False):
    """ Compute the gradient of the total cost wrt an assignment when using the given cost matrix.
    """
    # dim=-2 and 2 is an operation over rows
    # dim=-1 and 3 is and operation over columns

    # Input shapes:
    # assignment :: Tensor(n, 1, i, k) or Tensor(n, 1, p, q)
    # cost_matrix :: Tensor(n, 1, i, j) or Tensor(n, 1, p, j)

    # compute the right term first
    zero_padding = assignment.new(assignment.size()[:-1] + (1,)).fill_(0)
    if not local:
        cumu = assignment.cumsum(dim=-1)
        reverse_cumu = assignment.flip([3]).cumsum(dim=-1).flip([3])
    else:
        cumu = assignment
        reverse_cumu = assignment
    k_lt_q = torch.cat([zero_padding, cumu], dim=-1)[..., :-1]
    k_gt_q = torch.cat([reverse_cumu, zero_padding], dim=-1)[..., 1:]

    weight = k_gt_q - k_lt_q  # :: Tensor(n, 1, j, q)
    weight = weight.squeeze(dim=1)  # :: Tensor(n, p, j)
    cost_matrix = cost_matrix.squeeze(dim=1)  # :: Tensor(n, j, q)
    grad = 2 * torch.bmm(cost_matrix, weight)  # :: Tensor(n, p, q)
    return grad.unsqueeze(dim=1)  # :: Tensor(n, 1, p, q)


def sinkhorn_2d(x, steps=1, temp=1):
    # flatten the 2d dim to one for normalisation
    original_size = x.size()
    x = x.view(*original_size[:-2], -1)

    x = sinkhorn(x, steps=steps, temp=temp)

    # undo the flattening
    x = x.view(original_size)
    return x


def sinkhorn(x, steps=1, temp=1):
    """ Apply the Sinkhorn operator with an exp in front on the last two dimensions.
    """
    x = F.softmax(x / temp, dim=-1)
    for _ in range(steps):
        x = x / x.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        x = x / x.sum(dim=-2, keepdim=True).clamp(min=1e-12)
    return x


def outer(a, b=None):
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b


class Comparator(nn.Module):
    def __init__(self, skew):
        super().__init__()
        self.skew = skew

    def forward(self, x):
        a, b = outer(x)
        x = torch.cat([a, b], dim=1)
        y = torch.cat([b, a], dim=1)
        skew = self.skew(x) - self.skew(y)
        return skew


class LinearAssign(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.f = model

    def forward(self, x):
        x = self.f(x)
        return x.unsqueeze(1).transpose(-1, -2)


class Conv1(nn.Module):
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, 1, **kwargs)

    def forward(self, x):
        n, c, *s = x.size()
        x = x.view(n, c, -1)
        x = self.conv(x)
        x = x.view(n, -1, *s)
        return x
