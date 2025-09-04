import React, { useState } from 'react';

// Static demo data derived from datasets under data/
const lineLists = {
  ADAS: {
    values: [1.0, 0.8, 0.5],
  },
  CHIANTI: {
    values: [1.1, 0.9, 0.4],
  },
};

const crossSections = {
  'LXCat': {
    values: [1e-20, 2e-20, 3e-20],
  },
  'LXCat Alt': {
    values: [1.5e-20, 2.5e-20, 3.5e-20],
  },
};

export default function DatasetSwap() {
  const [line, setLine] = useState('ADAS');
  const [xs, setXs] = useState('LXCat');

  const metric = () => {
    const l = lineLists[line].values.reduce((a, b) => a + b, 0) / lineLists[line].values.length;
    const x = crossSections[xs].values.reduce((a, b) => a + b, 0) / crossSections[xs].values.length;
    return (l * x).toExponential(2);
  };

  return (
    <div>
      <h4>Dataset Impact Demo</h4>
      <label>
        Line List
        <select value={line} onChange={(e) => setLine(e.target.value)}>
          {Object.keys(lineLists).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </label>
      <label>
        Cross Sections
        <select value={xs} onChange={(e) => setXs(e.target.value)}>
          {Object.keys(crossSections).map((k) => (
            <option key={k} value={k}>{k}</option>
          ))}
        </select>
      </label>
      <p>Combined metric: {metric()}</p>
    </div>
  );
}
