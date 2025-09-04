import React, { useState } from 'react';
import axios from 'axios';
import ProjectManager from './ProjectManager.jsx';

export default function App() {
  const [token, setToken] = useState('');
  const [config, setConfig] = useState('{}');
  const [runId, setRunId] = useState('');

  const [voltage, setVoltage] = useState(1.0);
  const [pressure, setPressure] = useState(0.1);
  const [projects, setProjects] = useState([]);


  const login = async (e) => {
    e.preventDefault();
    const form = new FormData(e.target);
    const { data } = await axios.post('/token', form);
    setToken(data.access_token);
  };

  const submitConfig = async (e) => {
    e.preventDefault();
    const cfg = JSON.parse(config);
    const { data } = await axios.post('/run', { config: cfg }, { headers: { Authorization: `Bearer ${token}` } });
    setRunId(data.run_id);
    setProjects((p) => [...p, { id: data.run_id, config: cfg }]);
  };

  const updateSimulation = async (v, p) => {
    if (!runId) return;
    await axios.post(`/update/${runId}`, { voltage: v, pressure: p }, { headers: { Authorization: `Bearer ${token}` } });
  };

  return (
    <div>
      {!token && (
        <form onSubmit={login}>
          <h3>Login</h3>
          <input name="username" placeholder="username" />
          <input name="password" type="password" placeholder="password" />
          <button type="submit">Login</button>
        </form>
      )}
      {token && (
        <>
          <form onSubmit={submitConfig}>
            <h3>Submit Config</h3>
            <textarea rows={10} cols={50} value={config} onChange={(e) => setConfig(e.target.value)} />
            <br />
            <button type="submit">Run</button>
          </form>
          <ProjectManager projects={projects} />
        </>
      )}
      {runId && (
        <div>
          <p>Submitted run: {runId}</p>
          <div>
            <label title="Adjust the driving voltage applied to the plasma sheath.">
              Voltage: {voltage.toFixed(2)} kV
              <input
                type="range"
                min="0"
                max="5"
                step="0.1"
                value={voltage}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  setVoltage(v);
                  updateSimulation(v, pressure);
                }}
              />
            </label>
            <details>
              <summary>What is this?</summary>
              Controls the electric potential driving the sheath evolution.
            </details>
          </div>
          <div>
            <label title="Set the background gas pressure used in the simulation.">
              Pressure: {pressure.toFixed(2)} bar
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={pressure}
                onChange={(e) => {
                  const p = parseFloat(e.target.value);
                  setPressure(p);
                  updateSimulation(voltage, p);
                }}
              />
            </label>
            <details>
              <summary>What is this?</summary>
              Represents ambient gas pressure; higher values damp the sheath faster.
            </details>
          </div>
        </div>
      )}
    </div>
  );
}
