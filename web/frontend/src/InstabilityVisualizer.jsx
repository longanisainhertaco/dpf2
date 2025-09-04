import React, { useEffect, useRef, useState } from 'react';

export default function InstabilityVisualizer() {
  const [points, setPoints] = useState([]);
  const svgRef = useRef(null);
  const canvasRef = useRef(null);
  const [phase, setPhase] = useState(0);

  const phaseNames = ['Breakdown', 'Rundown', 'Pinch', 'Afterglow'];

  useEffect(() => {
    const data = [];
    for (let t = 0; t <= 1; t += 0.05) {
      const amp = Math.sin(t * 6) * Math.exp(t * 2);
      data.push({ t, amp });
    }
    setPoints(data);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const fields = [
      (x, y) => ({ u: 0, v: -y }), // Breakdown
      (x, y) => ({ u: -y, v: x }), // Rundown
      (x, y) => ({ u: -x, v: -y }), // Pinch
      () => ({ u: 0, v: 0 }), // Afterglow
    ];
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'green';
      const field = fields[phase];
      for (let x = 20; x < canvas.width; x += 40) {
        for (let y = 20; y < canvas.height; y += 20) {
          const { u, v } = field(x - 100, y - 50);
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x + u * 10, y + v * 10);
          ctx.stroke();
        }
      }
      requestAnimationFrame(draw);
    };
    draw();
    const timer = setInterval(() => setPhase((p) => (p + 1) % 4), 1000);
    return () => clearInterval(timer);
  }, [phase]);

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
    <div
      className="overlay"
      title="Illustrates a growing instability amplitude and plasma motion"
    >
      <h4>Instability Growth</h4>
      <canvas
        ref={canvasRef}
        width="200"
        height="100"
        title="Live vector field for current phase"
      />
      <svg ref={svgRef} width="200" height="100">
        <polyline points={pts} stroke="red" fill="none" />
      </svg>
      <div>Phase: {phaseNames[phase]}</div>
      <button
        type="button"
        onClick={download}
        title="Download this visualization"
      >
        Download
      </button>
      <details>
        <summary>What is this?</summary>
        This synthetic plot depicts how an instability might grow during the
        afterglow phase. The vector field approximates plasma motion through
        breakdown, rundown, pinch, and afterglow stages.
      </details>
    </div>
  );
}
