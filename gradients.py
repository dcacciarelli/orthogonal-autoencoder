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
        grad = get_grads(torch.Tensor(elem[0]), model)
        grads.append(np.array(grad))

    # 3. Approximate the integral using the trapezoidal rule
    delta = 1 / steps
    integrated_grads = delta * np.sum((inp - baseline) * grads, axis=0)

    return integrated_grads, np.mean(grads, axis=0)


def get_integrated_gradients_mc(model, inp, baseline, samples=1000):

    # If baseline is not provided, start with an array of zeros
    if baseline is None:
        baseline = np.zeros(len(inp)).astype(np.float32)
    else:
        baseline = baseline

    baseline = np.array(baseline).reshape(-1, len(inp))
    inp = np.array(inp).reshape(-1, len(inp))

    # Generate random samples
    random_samples = np.random.uniform(0, 1, (samples, len(inp)))
    random_vectors = baseline + (inp - baseline) * random_samples

    # Compute gradients for each random vector
    grads = []
    for i in range(samples):
        grad = get_grads(torch.Tensor(random_vectors[i, :]), model)
        grads.append(np.array(grad))

    # Calculate the average gradient and multiply by the difference between the input and baseline
    integrated_grads = (inp - baseline) * np.mean(grads, axis=0)

    return integrated_grads.reshape(-1,), np.mean(grads, axis=0)


def get_integrated_gradients_rect(model, inp, baseline, steps=20):

    # If baseline is not provided, start with an array of zeros
    if baseline is None:
        baseline = np.zeros(len(inp)).astype(np.float32)
    else:
        baseline = baseline

    baseline = np.array(baseline).reshape(-1, len(inp))
    inp = np.array(inp).reshape(-1, len(inp))

    # Compute gradients for each midpoint
    # delta = 1 / rects
    # integrated_grads = np.zeros(len(inp))
    # for i in range(rects):
    #     midpoint = baseline + delta * (i + 0.5) * (inp - baseline)
    #     grad = get_grads(torch.Tensor(midpoint), model)
    #     integrated_grads += delta * (inp - baseline) * grad

    # 1. Interpolation: path between baseline and input
    path_inputs = [baseline + (i / steps) * (inp - baseline) for i in range(steps + 1)]

    # 2. Compute gradients
    grads = []
    for elem in path_inputs:
        grad = get_grads(torch.Tensor(elem[0]), model)
        grads.append(np.array(grad))

    # 3. Approximate the integral using the rectangular rule
    avg_grads = grads[0]
    for i in range(1, len(grads)):
        avg_grads += grads[i]
    avg_grads /= steps

    # 4. Calculate integrated gradients and return
    integrated_grads = ((inp - baseline) * avg_grads)[0]

    return integrated_grads, avg_grads

