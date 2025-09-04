import React, { useState, useEffect } from 'react';
import axios from 'axios';
import YieldPressureOverlay from './YieldPressureOverlay.jsx';
import EfficiencyCurveOverlay from './EfficiencyCurveOverlay.jsx';

export default function ProjectManager({ projects }) {
  const [activeId, setActiveId] = useState(null);
  const [results, setResults] = useState({});
  const active = projects.find((p) => p.id === activeId);

  useEffect(() => {
    if (!activeId) return undefined;
    const fetchResults = async () => {
      try {
        const { data } = await axios.get(`/sweep/${activeId}`);
        setResults((r) => ({ ...r, [activeId]: data }));
      } catch (err) {
        // ignore fetch errors
      }
    };
    fetchResults();

    const ws = new WebSocket(
      `${window.location.origin.replace('http', 'ws')}/ws/sweep/${activeId}`
    );
    ws.onmessage = (evt) => {
      const msg = JSON.parse(evt.data);
      if (msg.parameter !== undefined) {
        setResults((r) => ({
          ...r,
          [activeId]: {
            ...(r[activeId] || {}),
            [msg.parameter]: { yield: msg.yield, efficiency: msg.efficiency },
          },
        }));
      }
    };
    return () => ws.close();
  }, [activeId]);

  const activeResults = activeId ? results[activeId] || {} : {};

  return (
    <div>
      <h3>Projects</h3>
      {projects.length === 0 && <p>No projects submitted yet.</p>}
      <ul>
        {projects.map((p) => (
          <li key={p.id}>
            <button type="button" onClick={() => setActiveId(p.id)}>
              {p.id}
            </button>
          </li>
        ))}
      </ul>
      {active && (
        <div className="overlays">
          <YieldPressureOverlay data={activeResults} />
          <EfficiencyCurveOverlay data={activeResults} />
        </div>
      )}
    </div>
  );
}
