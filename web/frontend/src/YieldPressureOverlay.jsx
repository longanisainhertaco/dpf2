import React from 'react';

export default function YieldPressureOverlay({ data = [] }) {
  return (
    <div className="overlay" title="Shows how yield varies with pressure">
      <h4>Yield/Pressure Curve</h4>
      <svg width="200" height="100">
        <polyline
          points="0,100 50,80 100,60 150,40 200,20"
          stroke="blue"
          fill="none"
        />
      </svg>
    </div>
  );
}
