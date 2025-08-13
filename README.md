# Application Description:

## Inspiration
Drug discovery is a brutally large search problem. I wondered what if a system could “think” while it optimizes using quantum rigor for physics and a brain-like evaluator for taste. Quantum computing was such a great application, able to replicate responses of the brain and its response to different chemicals

## What I built
A quantum-classical loop for molecular design:

1) Quantum GAN (PennyLane) that proposes molecular fingerprints. This can be used to find the optimal chemical balances
2) VQE (Qiskit) estimates binding energies and convergence in real time. It's important to assess the effects that these drugs can have on the brain.
3) A brain-inspired evaluator (adaptive, noise-aware neural system) scores candidates with evolving preferences. It is a major portion of the project, allowing us to consider what responses we would have in real life. 
4) Node/WebSocket backend streams typed events (rewards, binding, synapses, molecules).
5) React dashboard renders charts, heatmaps, and 3D structures (3Dmol.js).

## How it works (core idea)
Instead of optimizing a single static metric, QBrainX co-evolves the generator, quantum binding energy, and brain-like reward:

Instead of optimizing a single static metric, QBrainX co-evolves three components: the GAN generator, quantum-estimated binding energy, and an adaptive brain-like reward.

E_bind: Binding energy estimated via Variational Quantum Eigensolver (VQE).

R_brain: Adaptive reward from the quantum brain evaluator.

λ terms: Coefficients that control how much each component influences the total objective.

## What I learned
Stable shapes (rewards, binding, synapses) made the frontend and backend click.
Hybrid orchestration patterns: spawn Python from Node, log to stderr, print JSON to stdout, stream deltas via WebSockets.
Quantum + ML integration: treating VQE as a differentiable penalty and the brain evaluator as a moving objective is powerful for steering generation.
Pragmatic visualization: 3Dmol.js with SMILES gives immediate structural insight; small UX details (e.g., background colors, fallbacks) matter. UX took much longer than anticipated and initially I didn't consider it to be that difficult to create relative to the other algorithms which is where I was mistaken.

## How I built it
Python: PennyLane (quantum generator), Qiskit (VQE), PyTorch (brain evaluator).
Node: Express + ws to launch Python and stream results.
React: Recharts for time series, custom heatmaps for synapses, 3Dmol.js for structures.
Challenges I faced
Dependency friction (e.g., heavy chem stacks like RDKit): I scoped to essentials and used SMILES + 3Dmol.js.
Cross-runtime reliability: making Python print only JSON on stdout, routing logs to stderr, and handling Windows Python launchers (python vs py).
Real-time UX: keeping charts smooth while running quantum simulations to which I added fallbacks and small payloads.
Multi-objective stability: tuning (\lambda_{\text{VQE}}) and (\lambda_{\text{brain}}) to avoid mode collapse or trivial minima.
Why it’s novel
Most pipelines optimize a fixed score. QBrainx co-evolves a generator with quantum energy and an adaptive, brain-like preference signal, turning search from “lowest number wins” into “physics-aligned, goal-aware discovery.” Essentially, QBrainX is novel because it uses quantum computing’s ability to simulate real brain processes like neuron entanglement and synaptic dynamics, which classical computers can’t replicate. With advanced quantum AI to discover new drugs, creating a platform that “thinks” and adapts like a brain in real time (rather than long, tedious experiments with real subjects at risk) is extremely incredible. It enables breakthroughs in neuroscience and pharmaceutical research by directly modeling nature’s quantum complexity. Classical computing, no matter how powerful, can only test molecules and simulate brain processes sequentially, hitting hard limits in complexity and scale. QBrainX uses quantum mechanics to explore vast molecular and neural possibilities in parallel, revealing solutions and patterns that were computationally unreachable before. QBrainX uses quantum superposition to evaluate millions of drug molecules simultaneously, quantum entanglement to model complex brain functions, and hybrid quantum-classical learning to refine results in real time.

# Getting Started with Create React App

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Available Scripts

In the project directory, you can run:

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
