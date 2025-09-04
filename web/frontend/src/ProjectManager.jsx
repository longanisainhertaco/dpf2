import React, { useState, useEffect } from 'react';
import axios from 'axios';
import YieldPressureOverlay from './YieldPressureOverlay.jsx';
import EfficiencyCurveOverlay from './EfficiencyCurveOverlay.jsx';

export default function ProjectManager({ projects = [] }) {
  const [configSets, setConfigSets] = useState(projects);
  const [selectedIds, setSelectedIds] = useState([]);
  const [results, setResults] = useState({});

  const [preset, setPreset] = useState('');
  const [cad, setCad] = useState(null);

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
        <input type="file" onChange={(e) => setCad(e.target.files[0])} />
        <button type="submit">Add</button>
      </form>

      {selectedIds.length > 0 && (
        <div className="overlays">
          <YieldPressureOverlay datasets={datasets} />
          <EfficiencyCurveOverlay datasets={datasets} />
        </div>
      )}
    </div>
  );
}
