import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import YieldPressureOverlay from './YieldPressureOverlay.jsx';
import EfficiencyCurveOverlay from './EfficiencyCurveOverlay.jsx';
import GeometryPresets from './GeometryPresets.jsx';
import help from './help.json';

export default function ProjectManager({ projects = [] }) {
  const [configSets, setConfigSets] = useState(projects);
  const [selectedIds, setSelectedIds] = useState([]);
  const [results, setResults] = useState({});

  const [preset, setPreset] = useState('');
  const [cad, setCad] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    setConfigSets(projects);
  }, [projects]);

  useEffect(() => {
    const sockets = [];
    selectedIds.forEach((id) => {
      if (!results[id]) {
        axios
          .get(`/sweep/${id}`)
          .then(({ data }) => setResults((r) => ({ ...r, [id]: data })))
          .catch(() => {});
      }
      const ws = new WebSocket(
        `${window.location.origin.replace('http', 'ws')}/ws/sweep/${id}`
      );
      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data);
        if (msg.parameter !== undefined) {
          setResults((r) => ({
            ...r,
            [id]: {
              ...(r[id] || {}),
              [msg.parameter]: {
                yield: msg.yield,
                efficiency: msg.efficiency,
                yield_per_shot: msg.yield_per_shot,
                yield_per_hour: msg.yield_per_hour,
                wall_plug_efficiency: msg.wall_plug_efficiency,
                lifetime_hours: msg.lifetime_hours,
              },
            },
          }));
        }
      };
      sockets.push(ws);
    });
    return () => sockets.forEach((s) => s.close());
  }, [selectedIds]);

  const toggleSelect = (id) => {
    setSelectedIds((ids) =>
      ids.includes(id) ? ids.filter((i) => i !== id) : [...ids, id]
    );
  };

  const addConfigSet = async (e) => {
    e.preventDefault();
    if (error) return;
    const form = new FormData();
    form.append('preset', preset);
    if (cad) form.append('cad', cad);
    try {
      const { data } = await axios.post('/projects', form);
      setConfigSets((c) => [...c, data]);
      setPreset('');
      setCad(null);
    } catch (err) {
      // ignore upload errors
    }
  };

  const importConfig = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const text = await file.text();
      const cfg = JSON.parse(text);
      const id = cfg.id || `import-${Date.now()}`;
      setConfigSets((c) => [...c, { id, config: cfg }]);
    } catch {
      setError('Invalid configuration file');
    }
  };

  const exportConfig = (project) => {
    if (!project?.config) return;
    const blob = new Blob([JSON.stringify(project.config, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${project.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const colors = ['blue', 'green', 'red', 'orange', 'purple'];
  const datasets = selectedIds.map((id, idx) => ({
    label: id,
    data: results[id] || {},
    color: colors[idx % colors.length],
  }));

  const metrics = useMemo(() => {
    const out = {};
    selectedIds.forEach((id) => {
      const res = results[id] || {};
      const vals = Object.values(res);
      if (vals.length) {
        out[id] = {
          maxYield: Math.max(...vals.map((v) => v.yield)),
          maxEfficiency: Math.max(...vals.map((v) => v.efficiency)),
        };
      }
    });
    return out;
  }, [results, selectedIds]);

  const handleFile = (e) => {
    const file = e.target.files[0];
    setError('');
    if (file) {
      const ok = /\.(step|stp|iges|igs|json)$/i.test(file.name);
      if (!ok) {
        setError('Unsupported geometry format');
        setCad(null);
        return;
      }
    }
    setCad(file);
  };

  return (
    <div>
      <h3>Configuration Sets</h3>
      {configSets.length === 0 && <p>No projects submitted yet.</p>}
      <ul>
        {configSets.map((p) => (
          <li key={p.id}>
            <label>
              <input
                type="checkbox"
                checked={selectedIds.includes(p.id)}
                onChange={() => toggleSelect(p.id)}
              />
              {p.id}
            </label>
            {p.config && (
              <button
                type="button"
                onClick={() => exportConfig(p)}
                title={help.project.export}
              >
                Export
              </button>
            )}
          </li>
        ))}
      </ul>

      <form onSubmit={addConfigSet}>
        <h4>Add Configuration Set</h4>
        <GeometryPresets onSelect={setPreset} />
        {preset && <p>Selected geometry: {preset}</p>}

        <input
          type="file"
          onChange={handleFile}
          title={help.project.geometryFile}
        />
        <button type="submit" title={help.project.newSet}>Add</button>
        <details>
          <summary>What is this?</summary>
          Drag a geometry preset into the drop zone or upload a CAD file to
          create a new configuration set. These settings can later be exported
          for sharing.
        </details>

      </form>

      <div>
        <h4>Import Configuration Set</h4>
        <input
          type="file"
          accept="application/json"
          onChange={importConfig}
          title={help.project.import}
        />
        <details>
          <summary>What is this?</summary>
          Import a previously exported configuration to compare or rerun
          simulations.
        </details>
      </div>

      {selectedIds.length > 1 && (
        <table className="comparison">
          <thead>
            <tr>
              <th>Project</th>
              <th>Max Yield</th>
              <th>Max Efficiency</th>
            </tr>
          </thead>
          <tbody>
            {selectedIds.map((id) => (
              <tr key={id}>
                <td>{id}</td>
                <td>
                  {metrics[id]?.maxYield !== undefined
                    ? metrics[id].maxYield.toFixed(3)
                    : '-'}
                </td>
                <td>
                  {metrics[id]?.maxEfficiency !== undefined
                    ? metrics[id].maxEfficiency.toFixed(3)
                    : '-'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selectedIds.length > 0 && (
        <div className="overlays">
          <YieldPressureOverlay datasets={datasets} />
          <EfficiencyCurveOverlay datasets={datasets} />
        </div>
      )}
    </div>
  );
}
