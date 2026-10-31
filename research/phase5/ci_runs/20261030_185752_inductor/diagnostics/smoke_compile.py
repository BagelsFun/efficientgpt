import time, torch
f = torch.compile(lambda x, y: torch.relu(x @ y) + 1.0)
x = torch.randn(64, 64, device='cuda'); y = torch.randn(64, 64, device='cuda')
t = time.perf_counter(); out = f(x, y); torch.cuda.synchronize()
print('tiny compile OK', round(time.perf_counter() - t, 1), 's, max err', float((out - (torch.relu(x @ y) + 1)).abs().max()))
