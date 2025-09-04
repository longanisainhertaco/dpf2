import React, { useMemo, useRef } from 'react';

export default function YieldPressureOverlay({ datasets = [] }) {
  const svgRef = useRef(null);

  const computePoints = (data) => {
    const entries = Object.entries(data);
    if (entries.length === 0) return '';
    const width = 200;
    const height = 100;
    const sorted = entries
      .map(([p, m]) => ({ param: parseFloat(p), value: m.yield }))
      .sort((a, b) => a.param - b.param);
    const minParam = sorted[0].param;
    const maxParam = sorted[sorted.length - 1].param;
    const maxValue = Math.max(...sorted.map((d) => d.value)) || 1;
    return sorted
      .map(({ param, value }) => {
        const x = ((param - minParam) / (maxParam - minParam || 1)) * width;
        const y = height - (value / maxValue) * height;
        return `${x},${y}`;
      })
      .join(' ');
  };

  const download = () => {
    const svg = svgRef.current;
    if (!svg) return;
    const blob = new Blob([svg.outerHTML], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'yield_pressure.svg';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="overlay">
      <h4>Yield/Pressure Curve</h4>
      <svg ref={svgRef} width="200" height="100">
        {datasets.map(({ label, data, color = 'blue' }) => {
          const pts = computePoints(data);
          return <polyline key={label} points={pts} stroke={color} fill="none" />;
        })}
      </svg>
      <button type="button" onClick={download}>
        Download
      </button>
    </div>
  );
}
