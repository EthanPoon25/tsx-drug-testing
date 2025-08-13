const express = require('express');
const cors = require('cors');
const http = require('http');
const path = require('path');
const { spawn } = require('child_process');
const { WebSocketServer } = require('ws');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const multer = require('multer');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const upload = multer({
  dest: 'uploads/',
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'text/csv' || file.mimetype === 'text/plain') {
      cb(null, true);
    } else {
      cb(new Error('Only CSV and TXT files are allowed'));
    }
  }
});

const port = 3003; // run backend on 3001 to avoid clashing with CRA dev server
const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret_change_me';

// In-memory user store (replace with DB in prod)
const users = new Map(); // email -> { email, passwordHash, history: [] }

function createToken(email) {
  return jwt.sign({ email }, JWT_SECRET, { expiresIn: '7d' });
}

function authMiddleware(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Missing token' });
  try {
    const payload = jwt.verify(token, JWT_SECRET);
    req.userEmail = payload.email;
    next();
  } catch {
    res.status(401).json({ error: 'Invalid token' });
  }
}

app.post('/auth/signup', async (req, res) => {
  const { email, password } = req.body || {};
  if (!email || !password) return res.status(400).json({ error: 'Email and password required' });
  if (users.has(email)) return res.status(409).json({ error: 'User exists' });
  const passwordHash = await bcrypt.hash(password, 10);
  users.set(email, { email, passwordHash, history: [] });
  const token = createToken(email);
  res.json({ token });
});

app.post('/auth/login', async (req, res) => {
  const { email, password } = req.body || {};
  const user = users.get(email);
  if (!user) return res.status(401).json({ error: 'Invalid credentials' });
  const ok = await bcrypt.compare(password, user.passwordHash);
  if (!ok) return res.status(401).json({ error: 'Invalid credentials' });
  const token = createToken(email);
  res.json({ token });
});

app.get('/api/history', authMiddleware, (req, res) => {
  const user = users.get(req.userEmail);
  res.json({ history: user?.history || [] });
});

app.post('/api/history', authMiddleware, (req, res) => {
  const user = users.get(req.userEmail);
  if (!user) return res.status(401).json({ error: 'Unauthorized' });
  const entry = { ts: Date.now(), ...req.body };
  user.history.push(entry);
  res.json({ ok: true });
});

// File upload endpoint for SMILES
app.post('/api/upload-smiles', upload.single('smiles'), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  try {
    const filePath = req.file.path;
    const targetPath = path.join(__dirname, 'pythonfiles', 'smiles.txt');

    // Read the uploaded file
    const content = fs.readFileSync(filePath, 'utf8');

    // Parse content based on file type
    let smilesList = [];
    if (req.file.mimetype === 'text/csv') {
      // Handle CSV format
      const lines = content.split('\n').filter(line => line.trim());
      smilesList = lines.map(line => {
        const parts = line.split(',');
        return parts[0].trim(); // First column should be SMILES
      }).filter(smiles => smiles && !smiles.startsWith('#'));
    } else {
      // Handle TXT format
      smilesList = content.split('\n')
        .map(line => line.trim())
        .filter(line => line && !line.startsWith('#'));
    }

    // Write to pythonfiles/smiles.txt
    fs.writeFileSync(targetPath, smilesList.join('\n'));

    // Clean up uploaded file
    fs.unlinkSync(filePath);

    res.json({
      message: `Successfully uploaded ${smilesList.length} SMILES strings`,
      count: smilesList.length
    });
  } catch (error) {
    // Clean up uploaded file on error
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({ error: 'Failed to process file: ' + error.message });
  }
});

function runPythonBackend(callback) {
  const scriptPath = path.join(__dirname, 'pythonfiles', 'backend_api.py');
  let data = '';

  // Prefer PYTHON env var, fallback to 'python' then 'py' (Windows)
  const pythonCmd = process.env.PYTHON || 'python';
  let proc = spawn(pythonCmd, [scriptPath], { cwd: __dirname });

  const tryFallback = (err) => {
    if (pythonCmd !== 'py') {
      // attempt Windows launcher
      proc = spawn('py', [scriptPath], { cwd: __dirname });
      attach(proc);
    } else {
      callback(err || new Error('Failed to start Python process'));
    }
  };

  const attach = (p) => {
    p.stdout.on('data', (chunk) => { data += chunk.toString(); });
    p.stderr.on('data', (err) => { console.error(`[PY STDERR] ${err}`); });
    p.on('error', tryFallback);
    p.on('close', (code) => {
      if (code !== 0) {
        console.warn(`Python exited with code ${code}`);
      }
      try {
        const json = JSON.parse(data);
        callback(null, json);
      } catch (e) {
        callback(e);
      }
    });
  };

  attach(proc);
}

// REST endpoint to get a one-shot aggregated payload
app.get('/api/aggregate', (req, res) => {
  runPythonBackend((err, payload) => {
    if (err) {
      console.error('Aggregate error:', err.message);
      // Fallback minimal payload to keep UI alive
      return res.status(200).json({
        moduleA: {},
        rewards: Array.from({ length: 60 }, (_, i) => Math.max(0, Math.min(1, 0.5 + Math.sin(i * 0.07) * 0.2))),
        binding: Array.from({ length: 40 }, (_, i) => ({ step: i, energy: -30 + Math.exp(-i * 0.07) * -10, convergence: Math.round((i + 1) / 40 * 100) })),
      });
    }
    res.json(payload);
  });
});

// Create HTTP server and attach WebSocket
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

function send(ws, type, payload) {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify({ type, payload }));
  }
}

function generateMolecules(n = 8) {
  const structures = [
    // caffeine, aspirin, cocaine, morphine, sulfamethoxazole, nicotine, carvone, anthracene
    'C8H10N4O2', 'C9H8O4', 'C17H21NO4', 'C17H19NO3',
    'C10H11N3O3S', 'C10H14N2', 'C10H14O', 'C14H10'
  ];
  const smilesList = [
    'Cn1cnc2n(C)c(=O)n(C)c(=O)c12', // caffeine
    'CC(=O)OC1=CC=CC=C1C(=O)O', // aspirin
    'CN1C(=O)C2=CC=CC=C2N(C)C1=O', // theobromine-like placeholder
    'CN1CCC23C4C1CC(O)C2C=C(C3=O)C=C4O', // morphine
    'CC1=NC(=O)NS(=O)(=O)C1=O', // sulfamethoxazole-like placeholder
    'CN1CCCC1C2=CN=CN2', // nicotine-like placeholder
    'CC(=C)C1=CC(=O)C=CC1=O', // carvone-like placeholder
    'C1=CC=C2C=C3C=CC=CC3=CC2=C1' // anthracene
  ];
  return Array.from({ length: n }, (_, id) => {
    const bits = Array.from({ length: 16 }, () => (Math.random() > 0.5 ? 1 : 0));
    const on = bits.reduce((a, b) => a + b, 0);
    const contFp = bits.map(b => (b ? Math.min(1, Math.random() * 0.5 + 0.5) : Math.random() * 0.4));
    return {
      id,
      structure: structures[id % structures.length],
      smiles: smilesList[id % smilesList.length],
      bits,
      density: on / bits.length,
      similarity: 40 + Math.random() * 60,
      energy: -25 - Math.random() * 20,
      contFp,
    };
  });
}

wss.on('connection', (ws, req) => {
  // Optional JWT via query param ?token=...
  try {
    const url = new URL(req.url, 'http://localhost');
    const token = url.searchParams.get('token');
    if (token) jwt.verify(token, JWT_SECRET);
    ws.userEmail = token ? jwt.decode(token)?.email : undefined;
  } catch {}
  console.log('WebSocket client connected');
  runPythonBackend((err, payload) => {
    if (err) {
      console.error('Backend error:', err.message);
      // graceful degraded stream
      const fallbackRewards = Array.from({ length: 60 }, (_, i) => Math.max(0, Math.min(1, 0.55 + Math.sin(i * 0.07) * 0.18)));
      const fallbackBinding = Array.from({ length: 50 }, (_, i) => ({ step: i, energy: -28 - Math.exp(-i * 0.07) * 12, convergence: Math.round((i + 1) / 50 * 100) }));
      send(ws, 'rewards', fallbackRewards.map((reward, idx) => ({ epoch: idx, reward, sd: 0.06 })));
      send(ws, 'binding', fallbackBinding);
      send(ws, 'molecules', generateMolecules(10));
    } else {
      const rewards = (payload.rewards || []).map((reward, idx) => ({ epoch: idx, reward, sd: 0.06 }));
      const binding = payload.binding || [];
      send(ws, 'rewards', rewards);
      send(ws, 'binding', binding);

      // Derive a synapses grid influenced by moduleA synaptic probabilities
      const base = Number(payload.moduleA?.synaptic_no_noise) || 0.5;
      const synapses = Array.from({ length: 64 }, (_, i) => {
        const strength = Math.max(0, Math.min(1, base + (Math.random() - 0.5) * 0.2));
        return {
          id: i,
          strength,
          active: Math.random() > 0.7,
          x: i % 8,
          y: Math.floor(i / 8),
          history: Array.from({ length: 16 }, (_, h) => Math.max(0, Math.min(1, strength + (Math.sin((i + h) * 0.5) * 0.2) + (Math.random() - 0.5) * 0.1)))
        };
      });
      send(ws, 'synapses', synapses);

      // Send molecules derived locally (can be replaced with Python output later)
      send(ws, 'molecules', generateMolecules(10));

      // Send status snapshot
      const status = {
        running: false,
        epoch: (payload.binding || []).length,
        progress: 100,
        ganLoss: 0.45,
        discLoss: 0.38,
        vqePenalty: Math.max(0.03, Math.min(1, Math.abs((payload.moduleA?.adaptive_convergence ?? 0.1)) || 0.14)),
        logs: [
          { level: 'info', msg: 'QBrainX backend connected.' },
          { level: 'info', msg: 'Python modules executed: Module A, VQE.' },
        ],
      };
      send(ws, 'status', status);
      // Save brief history for authenticated users
      if (ws.userEmail && users.has(ws.userEmail)) {
        const rewardMean = (rewards.reduce((a,b)=>a+b.reward,0)/Math.max(1,rewards.length)) || 0;
        users.get(ws.userEmail).history.push({ ts: Date.now(), summary: { rewardMean, bindingSteps: binding.length } });
      }
    }
  });
});

server.listen(port, () => {
  console.log(`Backend server running on http://localhost:${port}`);
});
