import numpy as np
import matplotlib.pyplot as plt

class HardwareGaussianRNG:
    def __init__(self, seed=42, lfsr_bits=16):
        self.state = seed & ((1 << lfsr_bits) - 1)
        self.lfsr_bits = lfsr_bits
        self.mask = (1 << lfsr_bits) - 1
        
    def lfsr_random(self):
        """16-bit LFSR generating uniform random [0, 1]"""
        # taps for 16-bit LFSR: 16, 14, 13, 11
        for _ in range(8): 
            bit = ((self.state >> 0) ^ (self.state >> 2) ^ 
                (self.state >> 3) ^ (self.state >> 5)) & 1
            self.state = (self.state >> 1) | (bit << (self.lfsr_bits - 1))
        return self.state / self.mask   # Uniform [0,1]

        
    
    def hardware_gaussian(self, mean=0, std=1, n_samples=12):
        """Generate Gaussian using CLT with LFSR"""
        uniform_sum = 0
        for _ in range(n_samples):
            uniform_sum += self.lfsr_random()
        
        # Convert to approximate N(0,1)
        gaussian = (uniform_sum - n_samples/2) / np.sqrt(n_samples/12)
        
        return mean + std * gaussian
    
    def generate_batch(self, size, mean=0, std=1, n_samples=12):
        """Generate batch of hardware Gaussian samples"""
        return np.array([self.hardware_gaussian(mean, std, n_samples) for _ in range(size)])

# Test the hardware Gaussian RNG
def test_hardware_gaussian():
    hw_rng = HardwareGaussianRNG(seed=42)
    
    n_samples = 10000
    hardware_gauss = hw_rng.generate_batch(n_samples, mean=0, std=1, n_samples=12)
    true_gauss = np.random.normal(0, 1, n_samples)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(hardware_gauss, bins=50, alpha=0.7, label='Hardware Gaussian', density=True)
    plt.hist(true_gauss, bins=50, alpha=0.7, label='True Gaussian', density=True)
    plt.title('Distribution Comparison')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    x = np.linspace(-4, 4, 100)
    plt.plot(x, 1/np.sqrt(2*np.pi)*np.exp(-x**2/2), 'r-', label='Theoretical N(0,1)')
    plt.hist(hardware_gauss, bins=50, alpha=0.7, label='Hardware', density=True)
    plt.title('Fit to Theoretical Gaussian')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical tests
    print(f"\nHardware Gaussian: mean = {hardware_gauss.mean():.4f}, std = {hardware_gauss.std():.4f}")
    print(f"True Gaussian:     mean = {true_gauss.mean():.4f}, std = {true_gauss.std():.4f}")
    print(f"Mean absolute error: {np.mean(np.abs(hardware_gauss - true_gauss)):.4f}")
    
    print("\n Effect of sample count")
    for n in [4, 8, 12, 16]:
        samples = hw_rng.generate_batch(1000, n_samples=n)
        print(f"{n} samples: mean={samples.mean():.3f}, std={samples.std():.3f}")

# Test neural network weight initialization
def test_nn_initialization():
    print("\nNeural network initialization")
    
    hw_rng = HardwareGaussianRNG(seed=123)
    
    # Xavier initialization parameters
    fan_in, fan_out = 3, 2  
    std = np.sqrt(2.0 / (fan_in + fan_out))
    
    # Hardware
    hardware_weights = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            hardware_weights[i,j] = hw_rng.hardware_gaussian(mean=0, std=std, n_samples=12)
    
    # Software
    numpy_weights = np.random.normal(0, std, (100, 100))
    
    print(f"Target std: {std:.4f}")
    print(f"Hardware weights std: {hardware_weights.std():.4f}")
    print(f"Numpy weights std: {numpy_weights.std():.4f}")
    print(f"Distribution similarity: {np.mean(np.abs(hardware_weights - numpy_weights)):.4f}")
    
    # Plot Comparison
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(hardware_weights.flatten(), bins=50, alpha=0.7, label='Hardware', density=True)
    plt.hist(numpy_weights.flatten(), bins=50, alpha=0.7, label='Numpy', density=True)
    plt.title('Weight Distribution Comparison')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(numpy_weights.flatten(), hardware_weights.flatten(), alpha=0.5)
    plt.plot([-3*std, 3*std], [-3*std, 3*std], 'r--')
    plt.title('Correlation: Hardware vs Numpy')
    plt.xlabel('Numpy Weights')
    plt.ylabel('Hardware Weights')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_hardware_gaussian()
    test_nn_initialization()