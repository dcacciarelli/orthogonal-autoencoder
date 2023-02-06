import torch
import numpy as np


def get_grads(x, trained_model):
    inp = x.unsqueeze(0).requires_grad_(True)
    output = trained_model.encoder(inp)  # trained model should have the encoding layers saved as "encoder"
    gradient = torch.autograd.grad(output.mean(), inp)[0][0].data.numpy()
    return gradient


# get_grads(x1[0,:]).shape

def get_integrated_gradients(model, inp, baseline, steps=20):

    # If baseline is not provided, start with an array of zeros
    if baseline is None:
        baseline = np.zeros(len(inp)).astype(np.float32)
    else:
        baseline = baseline

    baseline = np.array(baseline).reshape(-1, len(inp))
    inp = np.array(inp).reshape(-1, len(inp))

    # 1. Interpolation: path between baseline and input
    path_inputs = [baseline + (i / steps) * (inp - baseline) for i in range(steps + 1)]

    # 2. Compute gradients
    grads = []
    for elem in path_inputs:
        # print(elem)
        grad = get_grads(torch.Tensor(elem[0]), model)
        grads.append(grad[0])
    grads = torch.from_numpy(np.array(grads))

    # 3. Approximate the integral using the trapezoidal rule
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = torch.mean(grads, axis=0)

    # 4. Calculate integrated gradients and return
    integrated_grads = ((inp - baseline) * avg_grads.item())[0]
    return integrated_grads
