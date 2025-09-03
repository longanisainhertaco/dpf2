import React, { useState } from 'react';
import axios from 'axios';
import ProjectManager from './ProjectManager.jsx';

export default function App() {
  const [token, setToken] = useState('');
  const [config, setConfig] = useState('{}');
  const [runId, setRunId] = useState('');
  const [projects, setProjects] = useState([]);

  const login = async (e) => {
    e.preventDefault();
    const form = new FormData(e.target);
    const { data } = await axios.post('/token', form);
    setToken(data.access_token);
  };

  const submitConfig = async (e) => {
    e.preventDefault();
    const cfg = JSON.parse(config);
    const { data } = await axios.post('/run', { config: cfg }, { headers: { Authorization: `Bearer ${token}` } });
    setRunId(data.run_id);
    setProjects((p) => [...p, { id: data.run_id, config: cfg }]);
  };

  return (
    <div>
      {!token && (
        <form onSubmit={login}>
          <h3>Login</h3>
          <input name="username" placeholder="username" />
          <input name="password" type="password" placeholder="password" />
          <button type="submit">Login</button>
        </form>
      )}
      {token && (
        <>
          <form onSubmit={submitConfig}>
            <h3>Submit Config</h3>
            <textarea rows={10} cols={50} value={config} onChange={(e) => setConfig(e.target.value)} />
            <br />
            <button type="submit">Run</button>
          </form>
          <ProjectManager projects={projects} />
        </>
      )}
      {runId && <p>Submitted run: {runId}</p>}
    </div>
  );
}
