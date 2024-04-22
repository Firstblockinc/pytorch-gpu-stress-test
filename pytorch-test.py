import torch
import time
import multiprocessing
import argparse

assert torch.cuda.is_available()
assert torch.cuda.device_count() > 0

def stress_test_gpu(device_id, duration_seconds):
    print(f"Stress testing GPU {device_id} for {duration_seconds} seconds...")
    start_time = time.time()
    end_time = start_time + duration_seconds

    torch.cuda.set_device(device_id)

    while time.time() < end_time:
        tensor = torch.randn(1000, 1000, device='cuda')
        result = torch.mm(tensor, tensor)

    print(f"GPU {device_id} stress test completed for {duration_seconds} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU Stress Test')
    parser.add_argument('--duration', type=int, default=20,
                        help='Duration of the stress test in seconds')
    args = parser.parse_args()

    num_devices = torch.cuda.device_count()

    processes = []
    for i in range(num_devices):
        ctx = multiprocessing.get_context('spawn')
        p = ctx.Process(target=stress_test_gpu, args=(i, args.duration))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All GPUs stress tests completed.")
