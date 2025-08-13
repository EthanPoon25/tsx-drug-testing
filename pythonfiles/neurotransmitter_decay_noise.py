"""
neurotransmitter_decay_noise.py

Advanced quantum simulation of neurotransmitter decay using amplitude damping noise.

Features:
- Parameterized shots and noise level (0 to 1)
- Uses Qiskit AerSimulator with noise model properly applied
- Outputs counts and firing probability
- Visualizes results via histograms and noise-vs-firing probability curve
- Modular and reusable for integration into bigger brain simulation
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_synaptic_circuit():
    """
    Create a 1-qubit quantum circuit representing a neuron in superposition (50% firing).
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)  # Hadamard creates superposition => probabilistic firing
    qc.measure(0, 0)
    logger.debug("Created synaptic circuit with Hadamard gate.")
    return qc

def create_noise_model(noise_prob: float):
    """
    Create an amplitude damping noise model to simulate neurotransmitter decay.

    Args:
        noise_prob (float): Amplitude damping probability (0 to 1).

    Returns:
        NoiseModel: Qiskit noise model with amplitude damping applied after H gate.
    """
    noise_model = NoiseModel()
    if noise_prob > 0:
        error = amplitude_damping_error(noise_prob)
        noise_model.add_all_qubit_quantum_error(error, ['h'])
        logger.debug(f"Amplitude damping noise with probability {noise_prob} added.")
    return noise_model

def run_synaptic_simulation(shots=1000, noise_prob=0.0, visualize=True):
    """
    Run the quantum simulation of synaptic transmission with optional noise.

    Args:
        shots (int): Number of measurement shots.
        noise_prob (float): Amplitude damping noise probability.
        visualize (bool): Plot histogram of results if True.

    Returns:
        counts (dict): Measurement outcome counts.
        firing_prob (float): Probability of 'firing' (measured '1').
    """
    qc = create_synaptic_circuit()
    simulator = AerSimulator()
    noise_model = create_noise_model(noise_prob)

    logger.info(f"Running simulation: shots={shots}, noise_prob={noise_prob}")

    if noise_prob > 0:
        job = simulator.run(qc, shots=shots, noise_model=noise_model)
    else:
        job = simulator.run(qc, shots=shots)

    result = job.result()
    counts = result.get_counts()

    firing_count = counts.get('1', 0)
    firing_prob = firing_count / shots

    if visualize:
        title = f"Synaptic Transmission (Amplitude Damping Noise={noise_prob})"
        plot_histogram(counts, title=title)
        plt.show()

    logger.info(f"Firing probability: {firing_prob:.3f}")
    return counts, firing_prob

def experiment_varying_noise(shots=1000, max_noise=0.9, steps=10):
    """
    Run simulations over increasing noise levels to observe effect on firing probability.

    Args:
        shots (int): Number of shots per run.
        max_noise (float): Maximum noise probability.
        steps (int): Number of noise increments.
    """
    noise_levels = np.linspace(0, max_noise, steps)
    firing_probs = []

    for noise in noise_levels:
        _, prob = run_synaptic_simulation(shots=shots, noise_prob=noise, visualize=False)
        firing_probs.append(prob)
        logger.info(f"Noise {noise:.2f}: firing probability {prob:.3f}")

    plt.figure(figsize=(8,5))
    plt.plot(noise_levels, firing_probs, marker='o', linestyle='-')
    plt.title("Effect of Synaptic Noise on Firing Probability")
    plt.xlabel("Amplitude Damping Noise Probability")
    plt.ylabel("Firing Probability")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example: run a simple noisy synaptic simulation
    run_synaptic_simulation()

    # Run full noise-varying experiment
    experiment_varying_noise()
