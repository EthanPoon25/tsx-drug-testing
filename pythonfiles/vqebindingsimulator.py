import pennylane as qml
from pennylane import numpy as np
from typing import List, Tuple
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VQEBindingSimulator:
    def __init__(self, n_qubits: int = 8):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.history = {"energies": []}

        @qml.qnode(self.dev)
        def vqe_circuit(params, hamiltonian):
            self.ansatz_circuit(params)
            return qml.expval(hamiltonian)

        self.vqe_circuit = vqe_circuit

    def ansatz_circuit(self, params: np.ndarray):
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        for i in range(0, self.n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(1, self.n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

    def simulate_binding_energy(self, hamiltonian: qml.Hamiltonian, max_steps: int = 100, step_size: float = 0.2):
        params = np.random.random(self.n_qubits)

        def cost_fn(p):
            return self.vqe_circuit(p, hamiltonian)

        logger.info("Starting VQE optimization...")
        opt = qml.GradientDescentOptimizer(stepsize=step_size)

        self.history["energies"] = []  # reset history

        for i in range(max_steps):
            params = opt.step(cost_fn, params)
            energy = cost_fn(params)
            self.history["energies"].append(energy)
            logger.info(f"Step {i + 1}: Energy = {energy}")

        return energy

    def construct_hamiltonian(self, z_terms: List[Tuple[float, int]]) -> qml.Hamiltonian:
        coeffs = []
        ops = []
        for coeff, wire in z_terms:
            coeffs.append(coeff)
            ops.append(qml.PauliZ(wire))
        return qml.Hamiltonian(coeffs, ops)

    def plot_energy_convergence(self):
        if not self.history["energies"]:
            print("No energy history to plot.")
            return
        plt.plot(self.history["energies"], marker='o')
        plt.title("VQE Energy Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("Energy (Hartree)")
        plt.grid(True)
        plt.show()

    def compute_binding_affinity(self, ground_energy, n_electrons):
        reference_energy = -n_electrons * 0.5
        binding_affinity = ground_energy - reference_energy
        return binding_affinity


if __name__ == "__main__":
    sim = VQEBindingSimulator(n_qubits=4)

    # Example Hamiltonian: H = 0.5 * Z0 + 0.7 * Z1
    H = sim.construct_hamiltonian([(0.5, 0), (0.7, 1)])

    final_energy = sim.simulate_binding_energy(H, max_steps=20)
    print("Final binding energy estimate:", final_energy)

    # For the binding affinity, you need to define n_electrons
    n_electrons = 4  # Example, set according to your system
    binding_affinity = sim.compute_binding_affinity(final_energy, n_electrons)
    print(f"Estimated binding affinity: {binding_affinity}")

    sim.plot_energy_convergence()
