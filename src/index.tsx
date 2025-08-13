// Optional: Provided for convenience if you export this as a CRA/Vite project.
// Not used by the v0 preview. In CRA/Vite, import './app.css' inside app.js.
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./app"; // TSX file, no .tsx extension needed

const container = document.getElementById("root");
if (!container) throw new Error("Root element not found");

const root = createRoot(container);
root.render(<App />);

