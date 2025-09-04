import React, { useEffect, useRef } from 'react';

/**
 * Displays a mock real-time overlay of sheath position, J×B vectors,
 * and a forming beam. The visual is synthetic and intended as a
 * development placeholder.
 */
export default function SheathBeamOverlay() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let t = 0;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Sheath position as an oscillating cyan circle
      const r = 30 + 10 * Math.sin(t);
      ctx.strokeStyle = 'cyan';
      ctx.beginPath();
      ctx.arc(canvas.width / 2, canvas.height / 2, r, 0, Math.PI * 2);
      ctx.stroke();

      // J×B vectors as orange arrows around the sheath edge
      ctx.strokeStyle = 'orange';
      for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 4) {
        const x = canvas.width / 2 + Math.cos(angle) * r;
        const y = canvas.height / 2 + Math.sin(angle) * r;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + 10 * Math.cos(angle + Math.PI / 2), y + 10 * Math.sin(angle + Math.PI / 2));
        ctx.stroke();
      }

      // Beam formation as a magenta line extending from the center
      ctx.strokeStyle = 'magenta';
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, canvas.height / 2);
      ctx.lineTo(canvas.width / 2, canvas.height / 2 - 40 - 5 * t);
      ctx.stroke();

      t += 0.05;
      requestAnimationFrame(draw);
    };
    draw();
  }, []);

  return (
    <div className="overlay" title="Shows sheath edge, J×B drift, and beam emergence">
      <h4>Sheath & Beam</h4>
      <canvas ref={canvasRef} width="200" height="200" title="Live overlay canvas" />
      <details>
        <summary>What is this?</summary>
        The cyan circle estimates the sheath boundary, orange arrows depict J×B drift,
        and the magenta line illustrates beam formation. Data is synthesized for demo
        purposes.
      </details>
    </div>
  );
}
