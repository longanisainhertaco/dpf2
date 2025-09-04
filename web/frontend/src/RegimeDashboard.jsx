import React, { useEffect, useState } from 'react';

export default function RegimeDashboard() {
  const [data, setData] = useState([]);

  useEffect(() => {
    const ws = new WebSocket(
      `${window.location.origin.replace('http', 'ws')}/ws/regime`
    );
    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        setData((d) => [...d, msg]);
      } catch {
        /* ignore malformed messages */
      }
    };
    return () => ws.close();
  }, []);

  const limits = {
    S: 1,
    beta: 1,
    M_A: 1,
    R_m: 1,
    K_n: 0.1,
    omega_ce_tau_e: 1,
  };

  const latest = data[data.length - 1] || {};
  const check = (key) => {
    const val = latest[key];
    if (val === undefined) return '';
    if (['beta', 'M_A', 'K_n'].includes(key)) {
      return val > limits[key] ? 'violation' : '';
    }
    return val < limits[key] ? 'violation' : '';
  };

  return (
    <div className="overlay" title="Tracks plasma regime parameters">
      <h4>Regime Dashboard</h4>
      <table>
        <tbody>
          {Object.keys(limits).map((k) => (
            <tr key={k} className={check(k)}>
              <td>{k}</td>
              <td>{latest[k] !== undefined ? latest[k].toFixed(3) : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

