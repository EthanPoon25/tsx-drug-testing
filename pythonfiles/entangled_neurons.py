
"""
entangled_neurons.py
Simulates a pair of "entangled neurons" using quantum circuits.
Features:
- Creates entanglement between two qubits to mimic coupled neuron behavior.
- Adds configurable quantum noise (amplitude damping) to simulate synaptic degradation.
- Supports adaptive rotation gates to simulate neuroplasticity by adjusting connection strength.
- Provides measurement outcomes and plots histograms for visualization.
Dependencies:
- qiskit
- qiskit_aer
- matplotlib
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

def build_entangled_neurons(theta=0.5, noise_level=0.0):
    """
    Build a quantum circuit simulating two entangled neurons with adaptive coupling and noise.
    Args:
        theta (float): Parameter for Ry rotation on neuron 1 to simulate synaptic strength/plasticity (radians).
        noise_level (float): Amplitude damping error probability (0.0 to 1.0) representing synaptic decay.
    Returns:
        QuantumCircuit: The constructed quantum circuit.
        NoiseModel or None: Noise model object if noise_level > 0, else None.
    """
    qc = QuantumCircuit(2, 2)
    # Prepare neuron 0 in superposition (firing probabilistically)
    qc.h(0)
    # Adaptive synaptic connection strength modeled by Ry rotation on neuron 1
    qc.ry(theta, 1)
    # Create entanglement between neuron 0 and neuron 1 (simulate coupled firing)
    qc.cx(0, 1)
    # Measurement of both "neurons"
    qc.measure([0, 1], [0, 1])
    # Create noise model if noise_level specified
    noise_model = None
    if noise_level > 0:
        error = amplitude_damping_error(noise_level)
        noise_model = NoiseModel()
        # Apply noise to all qubits after Hadamard and Ry gates (simulate synaptic decay)
        noise_model.add_all_qubit_quantum_error(error, ['h', 'ry'])
    return qc, noise_model

def simulate_and_plot(theta=0.5, noise_level=0.0, shots=1000):
    """
    Runs the entangled neuron circuit on a simulator, optionally with noise,
    and plots the measurement results.
    Args:
        theta (float): Ry rotation angle for neuron 1.
        noise_level (float): Amplitude damping noise level.
        shots (int): Number of circuit executions to sample.
    """
    qc, noise_model = build_entangled_neurons(theta, noise_level)
    
    # Use AerSimulator with proper configuration
    simulator = AerSimulator()
    transpiled_circuit = transpile(qc, simulator)
    
    # Run the simulation using the modern approach
    if noise_model:
        job = simulator.run(transpiled_circuit, shots=shots, noise_model=noise_model)
    else:
        job = simulator.run(transpiled_circuit, shots=shots)
    
    result = job.result()
    counts = result.get_counts()
    
    title = f"Entangled Neurons (θ={theta:.2f} rad, Noise={noise_level*100:.1f}%)"
    plot_histogram(counts, title=title)
    plt.show()
    print("Measurement counts:", counts)

def learning_loop(initial_theta=0.1, noise_level=0.0, shots=1000, steps=15, target_prob=0.7):
    """
    Simulate adaptive learning by adjusting synaptic strength parameter theta
    to reach a target probability of both neurons firing together ('11' outcome).
    Args:
        initial_theta (float): Starting rotation angle for Ry gate.
        noise_level (float): Amplitude damping noise.
        shots (int): Number of measurements per step.
        steps (int): Number of learning iterations.
        target_prob (float): Desired probability of '11' firing pattern.
    Returns:
        List of (theta, prob_11) tuples recorded at each step.
    """
    import numpy as np
    
    theta = initial_theta
    history = []
    
    # Use AerSimulator with proper configuration
    simulator = AerSimulator()
    
    for step in range(steps):
        qc, noise_model = build_entangled_neurons(theta, noise_level)
        transpiled_circuit = transpile(qc, simulator)
        
        # Run the simulation using the modern approach
        if noise_model:
            job = simulator.run(transpiled_circuit, shots=shots, noise_model=noise_model)
        else:
            job = simulator.run(transpiled_circuit, shots=shots)
        
        counts = job.result().get_counts()
        
        # Probability of both neurons firing together (measured as '11')
        count_11 = counts.get('11', 0)
        prob_11 = count_11 / shots
        history.append((theta, prob_11))
        print(f"Step {step+1:02d}: θ={theta:.3f}, P('11')={prob_11:.3f}")
        
        # Simple feedback rule: adjust theta to move prob_11 toward target
        learning_rate = 0.1
        error = target_prob - prob_11
        theta += learning_rate * error
        
        # Keep theta within valid range [0, pi]
        theta = max(0, min(np.pi, theta))
    
    return history

def run_entangled_neurons_simulation(shots=1000, visualize=True):
    """
    Runs entangled neurons simulation with default parameters.
    Returns measurement counts and joint firing probability.
    """
    # Run simulation and plot if visualize
    if visualize:
        simulate_and_plot(theta=0.5, noise_level=0.1, shots=shots)
    
    # Also run without plotting to get counts for programmatic use
    qc, noise_model = build_entangled_neurons(theta=0.5, noise_level=0.1)
    from qiskit import transpile
    simulator = AerSimulator()
    transpiled = transpile(qc, simulator)
    if noise_model:
        job = simulator.run(transpiled, shots=shots, noise_model=noise_model)
    else:
        job = simulator.run(transpiled, shots=shots)
    result = job.result()
    counts = result.get_counts()
    return counts, counts.get('11', 0) / shots


if __name__ == "__main__":
    # Example usage
    print("Simulating entangled neurons with no noise, medium synaptic strength")
    simulate_and_plot(theta=0.5, noise_level=0.0)
    print("\nSimulating entangled neurons with 20% synaptic decay noise")
    simulate_and_plot(theta=0.5, noise_level=0.2)
    
    print("\n=== Learning loop: adaptive synaptic strength (theta) ===")
    results = learning_loop(initial_theta=0.1, noise_level=0.1, steps=15)
    
    # Plot results
    thetas, probs = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(thetas)+1), thetas, label="θ (synaptic strength)", marker='o')
    plt.plot(range(1, len(probs)+1), probs, label="P('11') - joint firing probability", marker='s')
    plt.xlabel("Learning Step")
    plt.ylabel("Value")
    plt.title("Adaptive Learning of Entangled Neurons")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()