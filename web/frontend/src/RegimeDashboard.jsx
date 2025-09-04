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

  const params = ['S', 'beta', 'M_A', 'R_m', 'K_n', 'omega_ce_tau_e'];
  const latest = data[data.length - 1] || {};
  const check = (key) =>
    latest.violations && latest.violations[key] ? 'violation' : '';

  return (
    <div className="overlay" title="Tracks plasma regime parameters">
      <h4>Regime Dashboard</h4>
      <table>
        <tbody>
          {params.map((k) => (
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

