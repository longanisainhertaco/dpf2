import React, { useEffect, useState } from 'react';

// --------------------------------------------------------------
// Helper to compute dimensionless parameters on the client.
// Parameters are expected to be in SI units.  ``L`` is the
// characteristic length scale of the device which defaults to 1 m
// for the lightweight UI calculations.
// --------------------------------------------------------------
function computeParams({
  n,
  T,
  B,
  v,
  eta,
  mfp,
  tau_e,
  L = 1.0,
}) {
  const mu0 = 4e-7 * Math.PI;
  const kB = 1.380649e-23;
  const mp = 1.67262192369e-27;
  const e = 1.602176634e-19;
  const me = 9.1093837015e-31;

  const sigma = eta > 0 ? 1.0 / eta : Infinity;
  const rho = n * mp;
  const vA = B / Math.sqrt(mu0 * rho);

  const S = mu0 * sigma * vA * L;
  const beta = (2 * mu0 * n * kB * T) / (B * B);
  const M_A = vA === 0 ? Infinity : v / vA;
  const R_m = mu0 * sigma * v * L;
  const K_n = mfp / L;
  const omega_ce_tau_e = (e * B / me) * tau_e;

  return { S, beta, M_A, R_m, K_n, omega_ce_tau_e };
}

export default function RegimeDashboard() {
  // Keep full history for timeline plotting and export.
  const [history, setHistory] = useState([]);
  const [warning, setWarning] = useState('');

  // Thresholds defining regime validity for the current physics model.
  const thresholds = {
    S: 1.0,
    beta: 1.0,
    M_A: 1.0,
    R_m: 1.0,
    K_n: 0.1,
    omega_ce_tau_e: 1.0,
  };

  useEffect(() => {
    const ws = new WebSocket(
      `${window.location.origin.replace('http', 'ws')}/ws/regime`
    );
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        let params = {};
        if (
          [
            'S',
            'beta',
            'M_A',
            'R_m',
            'K_n',
            'omega_ce_tau_e',
          ].every((k) => k in msg)
        ) {
          params = msg; // already computed on backend
        } else {
          params = computeParams(msg);
        }
        const entry = { ...params, step: msg.step || history.length, engine: msg.engine };
        setHistory((d) => [...d, entry]);
      } catch {
        /* ignore malformed messages */
      }
    };
    return () => ws.close();
  }, []);

  // Update warning when regime violates thresholds for chosen engine.
  useEffect(() => {
    if (!history.length) return;
    const latest = history[history.length - 1];
    const violations = Object.entries(thresholds).some(([k, limit]) => {
      const val = latest[k];
      return k === 'beta' || k === 'M_A' || k === 'K_n'
        ? val > limit
        : val < limit;
    });
    if (violations && latest.engine) {
      setWarning(
        `Physics engine ${latest.engine} may be inconsistent with current regime`
      );
    } else {
      setWarning('');
    }
  }, [history]);

  const params = ['S', 'beta', 'M_A', 'R_m', 'K_n', 'omega_ce_tau_e'];
  const latest = history[history.length - 1] || {};

  const check = (key) => {
    const val = latest[key];
    const limit = thresholds[key];
    return key === 'beta' || key === 'M_A' || key === 'K_n'
      ? val > limit
        ? 'violation'
        : ''
      : val < limit
      ? 'violation'
      : '';
  };

  // Simple SVG plotting for each parameter.
  const Plot = ({ keyName }) => {
    const width = 100;
    const height = 40;
    const maxVal = thresholds[keyName] * 2;
    const points = history
      .map((h, i) => {
        const x = (i / Math.max(history.length - 1, 1)) * width;
        const y = height - (Math.min(h[keyName], maxVal) / maxVal) * height;
        return `${x},${y}`;
      })
      .join(' ');
    const th = height - (thresholds[keyName] / maxVal) * height;
    return (
      <svg width={width} height={height} className="plot">
        <polyline
          fill="none"
          stroke="blue"
          strokeWidth="1"
          points={points}
        />
        <line
          x1="0"
          y1={th}
          x2={width}
          y2={th}
          stroke="red"
          strokeDasharray="4"
        />
      </svg>
    );
  };

  const exportTimeline = () => {
    const blob = new Blob([JSON.stringify(history, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'regime_timeline.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="overlay" title="Tracks plasma regime parameters">
      <h4>Regime Dashboard</h4>
      {warning && <div className="warning">{warning}</div>}
      <table>
        <tbody>
          {params.map((k) => (
            <tr key={k} className={check(k)}>
              <td>{k}</td>
              <td>{latest[k] !== undefined ? latest[k].toFixed(3) : '-'}</td>
              <td>
                <Plot keyName={k} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button onClick={exportTimeline}>Export</button>
    </div>
  );
}


