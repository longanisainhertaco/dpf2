import React, { useMemo, useRef } from 'react';

export default function YieldPressureOverlay({ data = {} }) {
  const svgRef = useRef(null);
  const points = useMemo(() => {
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
  }, [data]);

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
    <div className="overlay" title="Shows how yield varies with pressure">
      <h4>Yield/Pressure Curve</h4>
      <svg ref={svgRef} width="200" height="100">
        {points && <polyline points={points} stroke="blue" fill="none" />}
      </svg>
      <button type="button" onClick={download}>
        Download
      </button>
    </div>
  );
}
