import React, { useState } from 'react';
import YieldPressureOverlay from './YieldPressureOverlay.jsx';
import EfficiencyCurveOverlay from './EfficiencyCurveOverlay.jsx';

export default function ProjectManager({ projects }) {
  const [activeId, setActiveId] = useState(null);
  const active = projects.find((p) => p.id === activeId);

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
          <YieldPressureOverlay data={active.results?.yieldPressure} />
          <EfficiencyCurveOverlay data={active.results?.efficiency} />
        </div>
      )}
    </div>
  );
}
