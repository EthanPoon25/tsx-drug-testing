import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptiveQuantumNeuron:
    def __init__(self, initial_theta=np.pi/4, learning_rate=0.05, shots=1000, noise_prob=0.0):
        """
        Args:
            initial_theta (float): Initial Ry rotation angle (synaptic strength).
            learning_rate (float): Step size for theta updates.
            shots (int): Number of circuit runs per measurement.
            noise_prob (float): Amplitude damping noise probability to simulate decay.
        """
        self.theta = initial_theta
        self.learning_rate = learning_rate
        self.shots = shots
        self.noise_prob = noise_prob
        self.simulator = AerSimulator()

    def create_noise_model(self):
        noise_model = NoiseModel()
        if self.noise_prob > 0:
            error = amplitude_damping_error(self.noise_prob)
            noise_model.add_all_qubit_quantum_error(error, ['ry'])
            logger.debug(f"Added amplitude damping noise with probability {self.noise_prob}.")
        return noise_model

    def build_circuit(self):
        qc = QuantumCircuit(1, 1)
        qc.ry(self.theta, 0)
        qc.measure(0, 0)
        return qc

    def run_circuit(self):
        qc = self.build_circuit()
        noise_model = self.create_noise_model() if self.noise_prob > 0 else None

        logger.debug(f"Running circuit with θ={self.theta:.3f} rad and noise={self.noise_prob}")

        if noise_model:
            job = self.simulator.run(qc, shots=self.shots, noise_model=noise_model)
        else:
            job = self.simulator.run(qc, shots=self.shots)

        result = job.result()
        counts = result.get_counts()
        return counts

    def get_firing_probability(self, counts):
        total = sum(counts.values())
        if total == 0:
            logger.warning("No shots recorded — total count is 0. Returning firing probability = 0.")
            return 0.0
        prob = counts.get('1', 0) / total
        # Clamp to [0, 1] just in case of numerical issues
        prob = min(1.0, max(0.0, prob))
        return prob

    def update_theta(self, firing_prob, target=0.7):
        error = target - firing_prob
        # Adaptive learning rate based on error magnitude
        if abs(error) > 0.3:
            # Large error: use full learning rate
            adjustment = self.learning_rate * error
        elif abs(error) > 0.1:
            # Medium error: use 80% learning rate
            adjustment = self.learning_rate * error * 0.8
        else:
            # Small error: use 50% learning rate to prevent oscillation
            adjustment = self.learning_rate * error * 0.5
        
        self.theta += adjustment
        self.theta = max(0, min(np.pi, self.theta))  # Clamp between 0 and π

    def simulate_learning(self, epochs=20, target=0.7, verbose=True):
        history = []
        for epoch in range(epochs):
            counts = self.run_circuit()
            firing_prob = self.get_firing_probability(counts)
            self.update_theta(firing_prob, target)
            history.append((epoch + 1, self.theta, firing_prob))
            if verbose:
                logger.info(f"Epoch {epoch+1:02d} | θ={self.theta:.3f} | Firing Prob={firing_prob:.3f}")
        return history


def plot_learning_curve(history):
    epochs, thetas, firing_probs = zip(*history)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Theta (rad)', color=color)
    ax1.plot(epochs, thetas, color=color, marker='o', label='Theta (synaptic strength)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, np.pi)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Firing Probability', color=color)
    ax2.plot(epochs, firing_probs, color=color, marker='s', label='Firing Probability')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1)

    # Add legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.title("Adaptive Quantum Neuron Learning Curve")
    plt.tight_layout()
    plt.show()

def run_adaptive_learning_simulation(epochs=30, noise_prob=0.1, learning_rate=0.05, shots=1000, target=0.7):
    """
    Run adaptive learning simulation and return firing probabilities only.
    This matches the expected interface in run_simulation.py
    """
    # Use a more aggressive learning rate for better convergence
    adjusted_learning_rate = min(learning_rate, 0.12)  # Increased further
    
    neuron = AdaptiveQuantumNeuron(
        initial_theta=np.pi/10,  # Start very low
        learning_rate=adjusted_learning_rate,
        shots=shots,
        noise_prob=noise_prob
    )
    history = neuron.simulate_learning(epochs=epochs, target=target, verbose=True)
    
    # Extract just the firing probabilities for compatibility with run_simulation.py
    firing_probs = [float(prob) for epoch, theta, prob in history]  # Ensure float conversion
    
    # Also plot the detailed learning curve
    plot_learning_curve(history)
    
    return firing_probs

def run_entangled_neurons_simulation(shots=1000, visualize=True):
    """
    Wrapper function to match the interface expected by run_simulation.py
    This simulates entangled neurons behavior using adaptive learning.
    """
    logger.info("Simulating entangled neurons using adaptive quantum approach...")
    
    # Create two adaptive neurons with different initial parameters
    neuron1 = AdaptiveQuantumNeuron(initial_theta=np.pi/6, noise_prob=0.1, shots=shots//2)
    neuron2 = AdaptiveQuantumNeuron(initial_theta=np.pi/3, noise_prob=0.1, shots=shots//2)
    
    # Run a few learning steps to simulate entanglement-like behavior
    history1 = neuron1.simulate_learning(epochs=5, target=0.6, verbose=False)
    history2 = neuron2.simulate_learning(epochs=5, target=0.8, verbose=False)
    
    # Simulate joint measurement outcomes
    counts1 = neuron1.run_circuit()
    counts2 = neuron2.run_circuit()
    
    # Create combined counts dictionary (simulating joint measurement)
    combined_counts = {
        '00': counts1.get('0', 0) * counts2.get('0', 0) // (shots//2),
        '01': counts1.get('0', 0) * counts2.get('1', 0) // (shots//2),
        '10': counts1.get('1', 0) * counts2.get('0', 0) // (shots//2),
        '11': counts1.get('1', 0) * counts2.get('1', 0) // (shots//2)
    }
    
    if visualize:
        plot_histogram(combined_counts, title="Simulated Entangled Neurons")
        plt.show()
    
    # Calculate joint firing probability
    total_counts = sum(combined_counts.values())
    joint_prob = combined_counts.get('11', 0) / max(total_counts, 1)
    
    return combined_counts, joint_prob

if __name__ == "__main__":
    neuron = AdaptiveQuantumNeuron(noise_prob=0.1)  # Add some decay noise
    history = neuron.simulate_learning(epochs=30)
    plot_learning_curve(history)