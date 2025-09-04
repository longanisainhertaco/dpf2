import React, { useMemo, useRef } from 'react';

// ``datasets`` is an array of objects describing a parameter sweep.  Each
// dataset has a ``data`` object whose keys are the sweep parameter values and
// whose values are the metrics computed on the Python side.  This component
// renders a simple polyline plot of the wall plug efficiency and also exposes
// a small text summary of the best performing shot.
export default function EfficiencyCurveOverlay({ datasets = [] }) {
  const svgRef = useRef(null);

  const summary = useMemo(() => {
    // Flatten all metric objects into a single list and pick the one with the
    // highest wall-plug efficiency.  ``datasets`` can be empty or have missing
    // ``data`` dictionaries, so we guard for that as well.
    const metrics = datasets.flatMap((d) => Object.values(d.data || {}));
    if (metrics.length === 0) return null;
    const best = metrics.reduce((a, b) => {
      const aEff = a.wall_plug_efficiency ?? a.efficiency ?? 0;
      const bEff = b.wall_plug_efficiency ?? b.efficiency ?? 0;
      return bEff > aEff ? b : a;
    });
    return {
      yieldShot: best.yield_per_shot ?? 0,
      yieldHour: best.yield_per_hour ?? 0,
      wallPlug: best.wall_plug_efficiency ?? best.efficiency ?? 0,
      lifetime: best.lifetime_hours ?? 0,
    };
  }, [datasets]);

  const computePoints = (data) => {
    const entries = Object.entries(data);
    if (entries.length === 0) return '';
    const width = 200;
    const height = 100;
    const sorted = entries
      .map(([p, m]) => ({
        param: parseFloat(p),
        value: m.wall_plug_efficiency ?? m.efficiency,
      }))
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
    a.download = 'efficiency_curve.svg';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="overlay" title="Displays a hypothetical efficiency curve">
      <h4>Efficiency Curve</h4>
      <svg ref={svgRef} width="200" height="100">
        {datasets.map(({ label, data, color = 'green' }) => {
          const pts = computePoints(data);
          return <polyline key={label} points={pts} stroke={color} fill="none" />;
        })}
      </svg>
      {summary && (
        <ul className="metrics">
          <li>Yield/shot: {summary.yieldShot.toLocaleString()}</li>
          <li>Yield/hour: {summary.yieldHour.toLocaleString()}</li>
          <li>Wall-plug eff.: {(summary.wallPlug * 100).toFixed(2)}%</li>
          <li>Lifetime: {summary.lifetime.toFixed(1)} h</li>
        </ul>
      )}
      <button type="button" onClick={download}>
        Download
      </button>
    </div>
  );
}
