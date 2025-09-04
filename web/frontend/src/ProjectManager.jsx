import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import YieldPressureOverlay from './YieldPressureOverlay.jsx';
import EfficiencyCurveOverlay from './EfficiencyCurveOverlay.jsx';

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
          </li>
        ))}
      </ul>

      <form onSubmit={addConfigSet}>
        <h4>Add Configuration Set</h4>
        <select value={preset} onChange={(e) => setPreset(e.target.value)}>
          <option value="">Geometry Preset</option>
          <option value="tapered">Tapered</option>
          <option value="hollow">Hollow</option>
          <option value="re-entrant">Re-entrant</option>
        </select>
        <input type="file" onChange={handleFile} />
        {error && <p className="error">{error}</p>}
        <button type="submit">Add</button>
      </form>

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
