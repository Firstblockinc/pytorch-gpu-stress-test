import torch
import time
import multiprocessing

# Check if CUDA is available and at least one GPU is present
assert torch.cuda.is_available()
assert torch.cuda.device_count() > 0

# Function to stress test a single GPU
def stress_test_gpu(device_id, duration_seconds):
    print(f"Stress testing GPU {device_id}...")
    start_time = time.time()
    end_time = start_time + duration_seconds

    # Set the current CUDA device for this process
    torch.cuda.set_device(device_id)

    while time.time() < end_time:
        # Create a random tensor on the GPU
        tensor = torch.randn(1000, 1000, device='cuda')

        # Perform a matrix multiplication on the GPU
        result = torch.mm(tensor, tensor)

    print(f"GPU {device_id} stress test completed for {duration_seconds} seconds.")

if __name__ == '__main__':
    # Get the number of available CUDA devices
    num_devices = torch.cuda.device_count()

    # Specify the duration for the stress test (in seconds)
    stress_test_duration = 20

    # Create a process for each GPU using 'spawn' method
    processes = []
    for i in range(num_devices):
        ctx = multiprocessing.get_context('spawn')
        p = ctx.Process(target=stress_test_gpu, args=(i, stress_test_duration))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print("All GPUs stress tests completed.")
