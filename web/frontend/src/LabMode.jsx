import React, { useState } from 'react';
import axios from 'axios';

export default function LabMode({ token }) {
  const [configs, setConfigs] = useState(['{}']);
  const [shotCounts, setShotCounts] = useState([1]);
  const [useJitter, setUseJitter] = useState(false);
  const [switchJitter, setSwitchJitter] = useState(0.0);
  const [pressureJitter, setPressureJitter] = useState(0.0);
  const [gasPuffTiming, setGasPuffTiming] = useState(0.0);

  const addRun = () => {
    setConfigs((c) => [...c, '{}']);
    setShotCounts((s) => [...s, 1]);
  };

  const updateConfig = (idx, value) => {
    setConfigs((c) => c.map((cfg, i) => (i === idx ? value : cfg)));
  };

  const updateShotCount = (idx, value) => {
    setShotCounts((s) => s.map((count, i) => (i === idx ? value : count)));
  };

  const exportBundle = async () => {
    const runs = [];
    configs.forEach((text, idx) => {
      const cfg = JSON.parse(text || '{}');
      if (useJitter) {
        cfg.experimental_variability = {
          pressure_jitter_pct: pressureJitter,
          trigger_jitter_ns: switchJitter,
        };
      }
      cfg.gas_puff_timing_ns = gasPuffTiming;
      for (let i = 0; i < shotCounts[idx]; i += 1) {
        runs.push({ ...cfg });
      }
    });
    const { data } = await axios.post(
      '/lab-mode/manifests',
      { runs },
      {
        headers: { Authorization: `Bearer ${token}` },
        responseType: 'blob',
      }
    );
    const url = window.URL.createObjectURL(new Blob([data]));
    const a = document.createElement('a');
    a.href = url;
    a.download = 'manifest_bundle.zip';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div>
      <h3>Lab Mode</h3>
      <div>
        <label>
          <input
            type="checkbox"
            checked={useJitter}
            onChange={(e) => setUseJitter(e.target.checked)}
          />
          Enable Jitter
        </label>
        {useJitter && (
          <div>
            <label>
              Switch Jitter (ns)
              <input
                type="number"
                value={switchJitter}
                onChange={(e) => setSwitchJitter(parseFloat(e.target.value))}
              />
            </label>
            <label>
              Pressure Jitter (%)
              <input
                type="number"
                value={pressureJitter}
                onChange={(e) => setPressureJitter(parseFloat(e.target.value))}
              />
            </label>
            <label>
              Gas Puff Timing (ns)
              <input
                type="number"
                value={gasPuffTiming}
                onChange={(e) => setGasPuffTiming(parseFloat(e.target.value))}
              />
            </label>
          </div>
        )}
      </div>
      {configs.map((cfg, idx) => (
        <div key={idx}>
          <h4>Run {idx + 1}</h4>
          <textarea
            rows={6}
            cols={40}
            value={cfg}
            onChange={(e) => updateConfig(idx, e.target.value)}
          />
          <div>
            <label>
              Shots
              <input
                type="number"
                min={1}
                value={shotCounts[idx]}
                onChange={(e) => updateShotCount(idx, parseInt(e.target.value, 10))}
              />
            </label>
          </div>
        </div>
      ))}
      <button type="button" onClick={addRun}>
        Add Run
      </button>
      <button type="button" onClick={exportBundle} disabled={!token}>
        Export Manifest Bundle
      </button>
      <div>
        <h4>Variability Summary</h4>
        <p>Total Shots: {shotCounts.reduce((a, b) => a + b, 0)}</p>
        {useJitter && (
          <ul>
            <li>Switch Jitter: {switchJitter} ns</li>
            <li>Pressure Jitter: {pressureJitter} %</li>
            <li>Gas Puff Timing: {gasPuffTiming} ns</li>
          </ul>
        )}
      </div>
    </div>
  );
}
