import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ProjectManager from './ProjectManager.jsx';
import InstabilityVisualizer from './InstabilityVisualizer.jsx';
import SheathBeamOverlay from './SheathBeamOverlay.jsx';
import GuidedLabs from './GuidedLabs.jsx';

export default function App() {
  const [token, setToken] = useState('');
  const [config, setConfig] = useState('{}');
  const [runId, setRunId] = useState('');
  const [projects, setProjects] = useState([]);

  const [voltage, setVoltage] = useState(1.0);
  const [pressure, setPressure] = useState(0.1);

  // jitter controls
  const [useJitter, setUseJitter] = useState(false);
  const [switchJitter, setSwitchJitter] = useState(0.0);
  const [pressureJitter, setPressureJitter] = useState(0.0);

  // batch mode toggle
  const [batchMode, setBatchMode] = useState(false);

  // load snapshot on startup
  useEffect(() => {
    const saved = localStorage.getItem('dpfSnapshot');
    if (saved) {
      try {
        const snap = JSON.parse(saved);
        if (snap.config) setConfig(snap.config);
        if (snap.voltage !== undefined) setVoltage(snap.voltage);
        if (snap.pressure !== undefined) setPressure(snap.pressure);
        if (snap.useJitter !== undefined) setUseJitter(snap.useJitter);
        if (snap.switchJitter !== undefined) setSwitchJitter(snap.switchJitter);
        if (snap.pressureJitter !== undefined) setPressureJitter(snap.pressureJitter);
      } catch {
        /* ignore parse errors */
      }
    }
  }, []);

  // auto-save snapshot to localStorage
  useEffect(() => {
    const snapshot = {
      config,
      voltage,
      pressure,
      useJitter,
      switchJitter,
      pressureJitter,
    };
    localStorage.setItem('dpfSnapshot', JSON.stringify(snapshot));
  }, [config, voltage, pressure, useJitter, switchJitter, pressureJitter]);


  const login = async (e) => {
    e.preventDefault();
    const form = new FormData(e.target);
    const { data } = await axios.post('/token', form);
    setToken(data.access_token);
  };

  const submitConfig = async (e) => {
    e.preventDefault();
    const cfgBase = JSON.parse(config);

    const runConfigs = batchMode ? cfgBase : [cfgBase];

    for (const cfg of runConfigs) {
      if (useJitter) {
        cfg.experimental_variability = cfg.experimental_variability || {};
        cfg.experimental_variability.pressure_jitter_pct = pressureJitter;
        cfg.experimental_variability.trigger_jitter_ns = switchJitter;
      }
      const { data } = await axios.post(
        '/run',
        { config: cfg },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setRunId(data.run_id);
      setProjects((p) => [...p, { id: data.run_id, config: cfg }]);
    }

    // automatically save snapshot locally for sharing
    exportSnapshot();
  };

  const updateSimulation = async (v, p) => {
    if (!runId) return;
    await axios.post(`/update/${runId}`, { voltage: v, pressure: p }, { headers: { Authorization: `Bearer ${token}` } });
  };

  const exportSnapshot = () => {
    const snapshot = {
      config: JSON.parse(config),
      voltage,
      pressure,
      jitter: useJitter
        ? { pressure_jitter_pct: pressureJitter, trigger_jitter_ns: switchJitter }
        : undefined,
    };
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'snapshot.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const importSnapshot = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (evt) => setConfig(evt.target.result);
    reader.readAsText(file);
  };

  return (
    <div>
      {!token && (
        <form onSubmit={login}>
          <h3>Login</h3>
          <input name="username" placeholder="username" title="Your sandbox username" />
          <input name="password" type="password" placeholder="password" title="Your sandbox password" />
          <button type="submit" title="Authenticate with the server">Login</button>
        </form>
      )}
      {token && (
        <>
          <form onSubmit={submitConfig}>
            <h3>Submit Config</h3>
            <textarea
              rows={10}
              cols={50}
              value={config}
              onChange={(e) => setConfig(e.target.value)}
              title="Paste a JSON configuration for the simulation or array for batch runs"
            />
            <details>
              <summary>What is this?</summary>
              Provide a complete JSON configuration describing your
              scenario. It will be sent to the server and can be
              exported later for sharing.
            </details>
            <div>
              <label>
                <input
                  type="checkbox"
                  checked={useJitter}
                  onChange={(e) => setUseJitter(e.target.checked)}
                  title="Enable jitter for switch timing and fill pressure"
                />
                Enable Jitter
              </label>
              {useJitter && (
                <div>
                  <label title="Timing jitter in nanoseconds">
                    Switch Jitter (ns)
                    <input
                      type="number"
                      value={switchJitter}
                      onChange={(e) => setSwitchJitter(parseFloat(e.target.value))}
                    />
                  </label>
                  <label title="Fill pressure jitter as percent">
                    Pressure Jitter (%)
                    <input
                      type="number"
                      value={pressureJitter}
                      onChange={(e) => setPressureJitter(parseFloat(e.target.value))}
                    />
                  </label>
                </div>
              )}
            </div>
            <div>
              <label>
                <input
                  type="checkbox"
                  checked={batchMode}
                  onChange={(e) => setBatchMode(e.target.checked)}
                  title="Interpret config as array and run all entries"
                />
                Batch Run Manifest
              </label>
            </div>
            <div>
              <input
                type="file"
                accept="application/json"
                onChange={importSnapshot}
                title="Import a configuration snapshot"
              />
              <button
                type="button"
                onClick={exportSnapshot}
                title="Export the current configuration snapshot"
              >
                Export Snapshot
              </button>
              <details>
                <summary>What is this?</summary>
                Use snapshots to save or load simulation setups for sharing.
              </details>
            </div>
            <br />
            <button type="submit" title="Start the simulation run">Run</button>
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
          <InstabilityVisualizer />
          <SheathBeamOverlay voltage={voltage} pressure={pressure} />
          <GuidedLabs setVoltage={setVoltage} setPressure={setPressure} />
        </div>
      )}
    </div>
  );
}
