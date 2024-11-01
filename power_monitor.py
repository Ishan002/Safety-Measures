import psutil
import time

# Constants (initial values, adjust as needed)
IDLE_POWER = 10  # Estimated idle power consumption in watts (including SSD)
CPU_LOAD_THRESHOLD = 15 # Assume CPU load above 50% indicates moderate CPU usage
GPU_LOAD_THRESHOLD = 30  # Assume GPU load above 30% indicates moderate GPU usage
SSD_POWER = 1  # Estimated power consumption of an SSD in watts
DISPLAY_POWER = 5  # Estimated power consumption of the display in watts
OTHER_COMPONENTS_POWER = 5  # Estimated power consumption of other components (RAM, Wi-Fi, etc.) in watts
UPDATE_INTERVAL = 5  # Seconds

def get_cpu_load():
    return psutil.cpu_percent(interval=1)

def get_gpu_load():
    # Function to get GPU load (if applicable)
    # Replace this with actual GPU load monitoring code if needed
    return 0  # Assuming 0% GPU load initially

def get_power_consumption(cpu_load, gpu_load):
    # Calculate power consumption based on component loads
    cpu_power = IDLE_POWER + (cpu_load - CPU_LOAD_THRESHOLD) * 0.5 if cpu_load > CPU_LOAD_THRESHOLD else IDLE_POWER
    gpu_power = (gpu_load - GPU_LOAD_THRESHOLD) * 0.5 if gpu_load > GPU_LOAD_THRESHOLD else 0
    total_power = cpu_power + gpu_power + SSD_POWER + DISPLAY_POWER + OTHER_COMPONENTS_POWER
    return total_power

try:
    while True:
        cpu_load = get_cpu_load()
        gpu_load = get_gpu_load()  # Modify or replace this function based on your GPU monitoring
        power_consumption = get_power_consumption(cpu_load, gpu_load)
        
        print(f"Current CPU Load: {cpu_load}% | Current GPU Load: {gpu_load}% | Estimated Power Consumption: {power_consumption} watts")

        time.sleep(UPDATE_INTERVAL)
except KeyboardInterrupt:
    pass
