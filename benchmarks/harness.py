import torch

def cuda_time(fn,warmup=10,reps=50):
    for _ in range (warmup):
        fn()
    torch.cuda.synchronize()

    times=[]
    for _ in range(reps):
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return {"median_ms": times[len(times)//2], "min_ms": times[0], "max_ms": times[-1]}
