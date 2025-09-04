import React, { useEffect, useRef, useState } from 'react';

export default function InstabilityVisualizer() {
  const [points, setPoints] = useState([]);
  const svgRef = useRef(null);

  useEffect(() => {
    const data = [];
    for (let t = 0; t <= 1; t += 0.05) {
      const amp = Math.sin(t * 6) * Math.exp(t * 2);
      data.push({ t, amp });
    }
    setPoints(data);
  }, []);

  const pts = points
    .map(({ t, amp }) => {
      const x = t * 200;
      const y = 100 - ((amp + 1) / 2) * 100;
      return `${x},${y}`;
    })
    .join(' ');

  const download = () => {
    const svg = svgRef.current;
    if (!svg) return;
    const blob = new Blob([svg.outerHTML], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'instability.svg';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="overlay" title="Illustrates a growing instability amplitude">
      <h4>Instability Growth</h4>
      <svg ref={svgRef} width="200" height="100">
        <polyline points={pts} stroke="red" fill="none" />
      </svg>
      <button type="button" onClick={download} title="Download this visualization">Download</button>
      <details>
        <summary>What is this?</summary>
        This synthetic plot depicts how an instability might grow during the
        afterglow phase. Real simulations can export similar data.
      </details>
    </div>
  );
}
