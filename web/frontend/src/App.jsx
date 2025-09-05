import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ProjectManager from './ProjectManager.jsx';
import InstabilityVisualizer from './InstabilityVisualizer.jsx';
import SheathBeamOverlay from './SheathBeamOverlay.jsx';
import RegimeDashboard from './RegimeDashboard.jsx';
import DatasetSwap from './DatasetSwap.jsx';

import QuickStartTutorial from './QuickStartTutorial.jsx';
import VoltagePressureSliders from './VoltagePressureSliders.jsx';


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
    const params = new URLSearchParams(window.location.search);
    const snapId = params.get('snap');
    if (snapId) {
      axios.get(`/snapshot/${snapId}`).then(({ data }) => {
        if (data.config) setConfig(JSON.stringify(data.config));
        if (data.voltage !== undefined) setVoltage(data.voltage);
        if (data.pressure !== undefined) setPressure(data.pressure);
        if (data.jitter) {
          setUseJitter(true);
          setPressureJitter(data.jitter.pressure_jitter_pct || 0);
          setSwitchJitter(data.jitter.trigger_jitter_ns || 0);
        }
      });
      return;
    }
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

  const shareSnapshot = async () => {
    const snapshot = {
      config: JSON.parse(config),
      voltage,
      pressure,
      jitter: useJitter
        ? { pressure_jitter_pct: pressureJitter, trigger_jitter_ns: switchJitter }
        : undefined,
    };
    const { data } = await axios.post(
      '/snapshot/save',
      { state: snapshot },
      { headers: { Authorization: `Bearer ${token}` } }
    );
    const link = `${window.location.origin}?snap=${data.id}`;
    try {
      await navigator.clipboard.writeText(link);
      alert('Snapshot link copied to clipboard');
    } catch {
      alert(link);
    }
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
          <input name="username" placeholder="username" title={help.login.username} />
          <input name="password" type="password" placeholder="password" title={help.login.password} />
          <button type="submit" title={help.login.submit}>Login</button>
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
              title={help.config.textarea}
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
                  title={help.config.jitter.toggle}
                />
                Enable Jitter
              </label>
              {useJitter && (
                <div>
                  <label title={help.config.jitter.switch}>
                    Switch Jitter (ns)
                    <input
                      type="number"
                      value={switchJitter}
                      onChange={(e) => setSwitchJitter(parseFloat(e.target.value))}
                    />
                  </label>
                  <label title={help.config.jitter.pressure}>
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
                  title={help.config.batch}
                />
                Batch Run Manifest
              </label>
            </div>
            <div>
              <input
                type="file"
                accept="application/json"
                onChange={importSnapshot}
                title={help.config.import}
              />
              <button
                type="button"
                onClick={exportSnapshot}
                title={help.config.export}
              >
                Export Snapshot
              </button>
              <details>
                <summary>What is this?</summary>
                Use snapshots to save or load simulation setups for sharing.
              </details>
            </div>
            <br />
            <button type="submit" title={help.config.run}>Run</button>
          </form>
          <ProjectManager projects={projects} />
        </>
      )}
      {runId && (
        <div>
          <p>Submitted run: {runId}</p>
          <VoltagePressureSliders
            voltage={voltage}
            pressure={pressure}
            setVoltage={setVoltage}
            setPressure={setPressure}
            onChange={updateSimulation}
          />
          <button type="button" onClick={shareSnapshot}>
            Share Scene
          </button>
          <InstabilityVisualizer />
          <SheathBeamOverlay voltage={voltage} pressure={pressure} />
          <RegimeDashboard />
          <DatasetSwap />
          <QuickStartTutorial
            setVoltage={setVoltage}
            setPressure={setPressure}
          />

        </div>
      )}
    </div>
  );
}
