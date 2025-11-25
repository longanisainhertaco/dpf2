import React, { useEffect, useMemo, useRef, useState } from 'react';

function MiniPlot({ title, series, color, threshold, unit }) {
  const width = 220;
  const height = 100;
  const margin = 10;
  const maxVal = useMemo(
    () => Math.max(threshold * 1.2, ...series.map((p) => p.value), 1),
    [series, threshold]
  );
  const points = series
    .map((p, i) => {
      const x = margin + (i / Math.max(series.length - 1, 1)) * (width - margin * 2);
      const y =
        height -
        margin -
        (Math.min(p.value, maxVal) / maxVal) *
          (height - margin * 2);
      return `${x},${y}`;
    })
    .join(' ');

  const thresholdY =
    height - margin - (threshold / maxVal) * (height - margin * 2);

  return (
    <div className="pane" aria-label={`${title} trend`}>
      <div className="pane-title">
        <strong>{title}</strong>
        <span className="unit">{unit}</span>
      </div>
      <svg width={width} height={height}>
        <polyline fill="none" stroke={color} strokeWidth="2" points={points} />
        <line
          x1={margin}
          y1={thresholdY}
          x2={width - margin}
          y2={thresholdY}
          stroke="red"
          strokeDasharray="4"
        />
      </svg>
      <div className="latest">{series.at(-1)?.value.toFixed(3) ?? '-'} {unit}</div>
    </div>
  );
}

export default function MultiPanePlot({ voltage, pressure, runId }) {
  const [series, setSeries] = useState([]);
  const startRef = useRef(Date.now());

  useEffect(() => {
    setSeries([]);
    startRef.current = Date.now();
  }, [runId]);

  useEffect(() => {
    const timer = setInterval(() => {
      const t = (Date.now() - startRef.current) / 1000;
      const sheathRadius = Math.max(0.05, 0.4 - pressure * 0.8 + 0.02 * Math.sin(t));
      const driveCurrent = voltage * (0.8 + 0.4 * Math.sin(t * 2)) * (1 + pressure * 0.5);
      const driveVoltage = voltage * (1.0 + 0.1 * Math.cos(t));

      // Rough plasma gating for the regime dashboard. The synthetic flow uses
      // pressure to control collisionality and voltage to set the magnetisation
      // knob.
      const lundquist = (driveCurrent / Math.max(pressure, 0.05)) * 0.01;
      const omegaTau = (driveVoltage * 0.02) / Math.max(pressure, 0.05);
      const regimeIndex = Math.log10(1 + lundquist) + Math.log10(1 + omegaTau);

      setSeries((prev) => {
        const next = [
          ...prev,
          {
            time: t,
            current: driveCurrent,
            voltage: driveVoltage,
            sheath: sheathRadius,
            regime: regimeIndex,
          },
        ];
        return next.slice(-120);
      });
    }, 500);
    return () => clearInterval(timer);
  }, [pressure, voltage]);

  const panes = useMemo(() => {
    return [
      {
        title: 'Discharge Current',
        color: '#3b82f6',
        threshold: 5,
        unit: 'kA',
        series: series.map((p) => ({ time: p.time, value: p.current })),
        callout:
          'Trends with both voltage drive and neutral density. Peaks should delay as pressure rises, reflecting slower rundown.',
      },
      {
        title: 'Bank Voltage',
        color: '#f59e0b',
        threshold: voltage * 0.9,
        unit: 'kV',
        series: series.map((p) => ({ time: p.time, value: p.voltage })),
        callout:
          'Mirrors the imposed drive. Use it alongside the current pane to visualize dI/dt and expected inductive sheath forcing.',
      },
      {
        title: 'Sheath Radius',
        color: '#10b981',
        threshold: 0.15,
        unit: 'm',
        series: series.map((p) => ({ time: p.time, value: p.sheath })),
        callout:
          'Shrinks as pressure and drive increase. When the radius crosses the red line the pinch overlay should show maximum compression.',
      },
      {
        title: 'Regime Gate',
        color: '#8b5cf6',
        threshold: 1.0,
        unit: 'S, ωτ',
        series: series.map((p) => ({ time: p.time, value: p.regime ?? 0 })),
        callout:
          'Combines Lundquist and magnetisation surrogates. Values above the dashed line indicate magnetised, collisionless sheath motion suited to tight pinches.',
      },
    ];
  }, [series, voltage]);

  return (
    <div className="overlay" title="Synchronized plots for electrical drive and sheath evolution">
      <h4>Multi-pane Timeline</h4>
      <p>
        Each panel streams synthetic diagnostics tied to the current slider
        values. Use them to relate slider tweaks to sheath motion and to what the
        tutorial steps highlight in the WebGL overlay.
      </p>
      <div className="pane-grid">
        {panes.map((pane) => (
          <div key={pane.title} className="pane-card">
            <MiniPlot
              title={pane.title}
              series={pane.series}
              color={pane.color}
              threshold={pane.threshold}
              unit={pane.unit}
            />
            <p className="callout">{pane.callout}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
