import json
import os
from typing import List

# Aggregate outputs from the available python modules into a single JSON payload

from vqebindingsimulator import VQEBindingSimulator
from adaptive_neuron_learning import AdaptiveQuantumNeuron
from synaptic_transmission import run_synaptic_simulation as run_synapse
try:
    from neurotransmitter_decay_noise import run_synaptic_simulation as run_decay
except Exception:
    run_decay = None

# Optional RDKit support
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    from rdkit import DataStructs
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


def get_adaptive_rewards(epochs: int = 60, target: float = 0.7) -> list:
    """Return a list of firing probabilities over epochs using the adaptive neuron.

    Uses the class directly to avoid plotting side-effects in helper functions.
    """
    neuron = AdaptiveQuantumNeuron(noise_prob=0.1, shots=800)
    history = neuron.simulate_learning(epochs=epochs, target=target, verbose=False)
    rewards = [float(prob) for _, __, prob in history]
    return rewards


def get_binding_series(max_steps: int = 40, n_qubits: int = 4) -> list:
    """Run a short VQE and export the energy history as a series for the UI."""
    sim = VQEBindingSimulator(n_qubits=n_qubits)
    # Simple Z terms for a lightweight demo Hamiltonian
    H = sim.construct_hamiltonian([(0.5, 0), (0.7, 1)])
    _ = sim.simulate_binding_energy(H, max_steps=max_steps, step_size=0.2)
    energies = [float(e) for e in sim.history.get("energies", [])]
    total = max(1, len(energies))
    series = [
        {
            "step": idx,
            "energy": energy,
            "convergence": int(100 * (idx + 1) / total),
        }
        for idx, energy in enumerate(energies)
    ]
    return series


def load_smiles_from_files() -> List[str]:
    candidates = [
        os.path.join(os.path.dirname(__file__), 'smiles.txt'),
        os.path.join(os.path.dirname(__file__), 'smiles.csv'),
        os.path.join(os.getcwd(), 'smiles.txt'),
        os.path.join(os.getcwd(), 'smiles.csv'),
    ]
    smiles: List[str] = []
    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        # support csv with first column as SMILES
                        if ',' in line:
                            parts = [p.strip() for p in line.split(',')]
                            if parts and parts[0]:
                                smiles.append(parts[0])
                        else:
                            smiles.append(line)
            except Exception:
                pass
            if smiles:
                break
    # default fallback if nothing provided
    if not smiles:
        smiles = [
            'Cn1cnc2n(C)c(=O)n(C)c(=O)c12',   # caffeine
            'CC(=O)OC1=CC=CC=C1C(=O)O',      # aspirin
            'CN1CCC23C4C1CC(O)C2C=C(C3=O)C=C4O',  # morphine (approx)
            'CN1CCCC1C2=CN=CN2',             # nicotine (approx)
        ]
    return smiles


def compute_bits_from_smiles(smi: str, n_bits: int = 16) -> List[int]:
    if RDKit_AVAILABLE:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [0] * n_bits
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        arr = [int(x) for x in fp.ToBitString()]
        return arr
    # fallback: hash-based pseudo fingerprint
    import random
    random.seed(hash(smi) & 0xffffffff)
    return [1 if random.random() > 0.5 else 0 for _ in range(n_bits)]


def tanimoto_similarity(bits_a: List[int], bits_b: List[int]) -> float:
    if RDKit_AVAILABLE:
        bv1 = DataStructs.CreateFromBitString(''.join(str(b) for b in bits_a))
        bv2 = DataStructs.CreateFromBitString(''.join(str(b) for b in bits_b))
        return float(DataStructs.FingerprintSimilarity(bv1, bv2))
    # fallback: Jaccard on lists
    a_on = {i for i, v in enumerate(bits_a) if v}
    b_on = {i for i, v in enumerate(bits_b) if v}
    if not a_on and not b_on:
        return 0.0
    return len(a_on & b_on) / max(1, len(a_on | b_on))


def compute_energy_for_bits(bits: List[int], sim: VQEBindingSimulator) -> float:
    # Map bit=1 → small Z term coefficient
    z_terms = []
    for i, b in enumerate(bits):
        if b:
            z_terms.append((0.5, i))
    if not z_terms:
        z_terms = [(0.1, 0)]
    H = sim.construct_hamiltonian(z_terms)
    energy = sim.simulate_binding_energy(H, max_steps=20, step_size=0.2)
    try:
        return float(energy)
    except Exception:
        return 0.0


def build_molecules_payload(n_qubits: int = 16, limit: int = 12) -> List[dict]:
    smiles_list = load_smiles_from_files()[:limit]
    sim = VQEBindingSimulator(n_qubits=n_qubits)
    molecules = []
    bitsets = []
    for smi in smiles_list:
        bits = compute_bits_from_smiles(smi, n_bits=n_qubits)
        bitsets.append(bits)
    # Precompute average similarity baseline
    avg_similarities = []
    for i, bi in enumerate(bitsets):
        sims = []
        for j, bj in enumerate(bitsets):
            if i == j:
                continue
            sims.append(tanimoto_similarity(bi, bj))
        avg_similarities.append(sum(sims) / max(1, len(sims)))

    for idx, smi in enumerate(smiles_list):
        bits = bitsets[idx]
        energy = compute_energy_for_bits(bits, sim)
        on = sum(bits)
        density = on / max(1, len(bits))
        # contFp: simple mapped floats
        import random
        random.seed(hash(smi) & 0xffffffff)
        cont_fp = [random.uniform(0.5, 1.0) if b else random.uniform(0.0, 0.4) for b in bits]
        structure = smi
        if RDKit_AVAILABLE:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                try:
                    structure = CalcMolFormula(mol)
                except Exception:
                    structure = smi

        molecules.append({
            'id': idx,
            'structure': structure,
            'smiles': smi,
            'bits': bits,
            'density': density,
            'similarity': float(avg_similarities[idx]) * 100.0,  # percentage for UI
            'energy': float(energy),
            'contFp': cont_fp,
        })
    return molecules


def main():
    # Build a lightweight Module A summary without any plotting
    try:
        _, prob_syn = run_synapse(shots=800, noise_prob=0.0, visualize=False)
    except Exception:
        prob_syn = 0.5

    try:
        if run_decay:
            _, prob_decay = run_decay(shots=800, noise_prob=0.3, visualize=False)
        else:
            _, prob_decay = run_synapse(shots=800, noise_prob=0.3, visualize=False)
    except Exception:
        prob_decay = 0.4

    # Adaptive neuron reward series (used as rewards in UI)
    rewards_series = get_adaptive_rewards(epochs=70)
    final_prob = float(rewards_series[-1]) if rewards_series else 0.0
    convergence = abs(final_prob - 0.7)

    # VQE binding series (used in UI)
    binding_series = get_binding_series(max_steps=50)

    # Build molecules from SMILES (RDKit if available, fallback otherwise)
    molecules = build_molecules_payload(n_qubits=16, limit=12)

    payload = {
        "moduleA": {
            "synaptic_no_noise": float(prob_syn),
            "synaptic_decay": float(prob_decay),
            "adaptive_final_prob": final_prob,
            "adaptive_convergence": float(convergence),
        },
        "rewards": rewards_series,
        "binding": binding_series,
        "molecules": molecules,
    }

    print(json.dumps(payload))


if __name__ == "__main__":
    main()


