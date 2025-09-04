import React from 'react';

export default function EfficiencyCurveOverlay({ data = [] }) {
  return (
    <div className="overlay" title="Displays a hypothetical efficiency curve">
      <h4>Efficiency Curve</h4>
      <svg width="200" height="100">
        <polyline
          points="0,90 50,70 100,55 150,45 200,40"
          stroke="green"
          fill="none"
        />
      </svg>
    </div>
  );
}
