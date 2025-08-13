"""
synaptic_transmission.py

Simulates probabilistic synaptic transmission using a quantum circuit.

This module models a single neuron synapse as a qubit in superposition,
representing the probabilistic nature of neurotransmitter release. 
Includes optional noise modeling to simulate synaptic fatigue or decay.

Features:
- Parameterized shots and noise level
- Supports amplitude damping noise channel to simulate decay
- Visualization of measurement outcomes (histogram)
- Returns raw counts and firing probability for further processing
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synaptic_circuit():
    """
    Create a quantum circuit simulating a neuron in superposition.

    Returns:
        QuantumCircuit: A 1-qubit circuit with Hadamard gate applied.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Superposition for probabilistic firing
    qc.measure(0, 0)
    logger.debug("Created synaptic circuit with Hadamard gate.")
    return qc

def create_noise_model(noise_prob: float):
    """
    Create an amplitude damping noise model to simulate neurotransmitter decay.

    Args:
        noise_prob (float): Probability of amplitude damping (0 to 1).

    Returns:
        NoiseModel: Qiskit noise model with amplitude damping error.
    """
    noise_model = NoiseModel()
    if noise_prob > 0:
        error = amplitude_damping_error(noise_prob)
        noise_model.add_all_qubit_quantum_error(error, ['h'])
        logger.debug(f"Added amplitude damping noise with probability {noise_prob}.")
    return noise_model

def run_synaptic_simulation(shots: int = 1000, noise_prob: float = 0.0, visualize: bool = True):
    """
    Run the synaptic transmission simulation.

    Args:
        shots (int): Number of measurement shots.
        noise_prob (float): Amplitude damping noise probability.
        visualize (bool): Whether to display a histogram plot.

    Returns:
        dict: Measurement counts (e.g., {'0': 520, '1': 480})
        float: Estimated probability of synapse firing ('1' outcome)
    """
    qc = create_synaptic_circuit()
    
    # Use AerSimulator with proper configuration
    simulator = AerSimulator()
    noise_model = create_noise_model(noise_prob)
    
    logger.info(f"Running simulation with {shots} shots and noise_prob={noise_prob}")

    # Run the simulation using the modern approach
    if noise_prob > 0:
        job = simulator.run(qc, shots=shots, noise_model=noise_model)
    else:
        job = simulator.run(qc, shots=shots)
    
    result = job.result()
    counts = result.get_counts()

    # Calculate probability of firing (measured '1')
    firing_count = counts.get('1', 0)
    firing_prob = firing_count / shots

    if visualize:
        title = f"Synaptic Transmission (Noise={noise_prob})"
        plot_histogram(counts, title=title)
        plt.show()

    logger.info(f"Simulation complete: firing probability = {firing_prob:.3f}")
    return counts, firing_prob

def experiment_varying_noise(shots: int = 1000, max_noise: float = 0.9, steps: int = 10):
    """
    Run simulations over a range of noise probabilities and plot firing probability.

    Args:
        shots (int): Number of shots per simulation.
        max_noise (float): Maximum noise probability.
        steps (int): Number of noise steps between 0 and max_noise.
    """
    import numpy as np

    noise_levels = np.linspace(0, max_noise, steps)
    firing_probs = []

    for noise in noise_levels:
        _, prob = run_synaptic_simulation(shots=shots, noise_prob=noise, visualize=False)
        firing_probs.append(prob)

    # Plot noise vs firing probability
    plt.figure(figsize=(8,5))
    plt.plot(noise_levels, firing_probs, marker='o', linestyle='-')
    plt.title("Effect of Synaptic Noise on Firing Probability")
    plt.xlabel("Amplitude Damping Noise Probability")
    plt.ylabel("Firing Probability")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Run a basic simulation with default params
    run_synaptic_simulation()

    # Run noise experiment to visualize impact of neurotransmitter decay
    experiment_varying_noise()