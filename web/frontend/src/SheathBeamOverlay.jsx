import React, { useEffect, useRef, useState } from 'react';

/**
 * Renders sheath evolution and J×B vectors using WebGL.
 * The sheath radius scales with voltage while arrow length
 * scales with pressure. Animation is synthetic for sandbox use.
 */
export default function SheathBeamOverlay({ voltage, pressure }) {
  const canvasRef = useRef(null);
  const voltageRef = useRef(voltage);
  const pressureRef = useRef(pressure);
  const [phase, setPhase] = useState(0);
  const phaseRef = useRef(0);
  const phaseNames = ['Breakdown', 'Rundown', 'Pinch', 'Afterglow'];

  useEffect(() => {
    voltageRef.current = voltage;
  }, [voltage]);

  useEffect(() => {
    pressureRef.current = pressure;
  }, [pressure]);

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  // Cycle through synthetic phases to demonstrate annotations
  useEffect(() => {
    const timer = setInterval(() => setPhase((p) => (p + 1) % 4), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const gl = canvas.getContext('webgl');
    if (!gl) return;

    // Vertex shader
    const vsSource = `
      attribute vec2 a_position;
      void main() {
        gl_Position = vec4(a_position, 0.0, 1.0);
      }
    `;
    // Fragment shader with solid color
    const fsSource = `
      precision mediump float;
      uniform vec4 u_color;
      void main() {
        gl_FragColor = u_color;
      }
    `;

    function compile(type, source) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return shader;
    }

    const vs = compile(gl.VERTEX_SHADER, vsSource);
    const fs = compile(gl.FRAGMENT_SHADER, fsSource);
    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    gl.useProgram(program);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    const colorLoc = gl.getUniformLocation(program, 'u_color');

    function drawShape(data, mode, color) {
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STREAM_DRAW);
      gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(positionLoc);
      gl.uniform4fv(colorLoc, color);
      gl.drawArrays(mode, 0, data.length / 2);
    }

    const render = () => {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);

      const v = voltageRef.current;
      const p = pressureRef.current;
      const radius = 0.2 + (v / 5) * 0.6; // scale radius with voltage

      // Build circle as triangle fan
      const circle = [0, 0];
      const segments = 32;
      for (let i = 0; i <= segments; i++) {
        const angle = (i / segments) * Math.PI * 2;
        circle.push(Math.cos(angle) * radius, Math.sin(angle) * radius);
      }
      drawShape(circle, gl.TRIANGLE_FAN, [0, 1, 1, 1]);

      // JxB arrows as line segments
      const arrows = [];
      const arrowLen = 0.05 + p * 0.15;
      for (let i = 0; i < 8; i++) {
        const angle = (i / 8) * Math.PI * 2;
        const x = Math.cos(angle) * radius;
        const y = Math.sin(angle) * radius;
        const dx = -Math.sin(angle) * arrowLen;
        const dy = Math.cos(angle) * arrowLen;
        arrows.push(x, y, x + dx, y + dy);
      }
      drawShape(arrows, gl.LINES, [1, 0.5, 0, 1]);

      // Vector field overlay depending on phase
      const field = [];
      const grid = 5;
      const spacing = 2 / (grid - 1); // normalized device coords
      for (let i = 0; i < grid; i++) {
        for (let j = 0; j < grid; j++) {
          const x = -1 + i * spacing;
          const y = -1 + j * spacing;
          let u = 0;
          let v = 0;
          switch (phaseRef.current) {
            case 0:
              u = x;
              v = y;
              break; // Breakdown: radial outward
            case 1:
              u = -y;
              v = x;
              break; // Rundown: azimuthal
            case 2:
              u = -x;
              v = -y;
              break; // Pinch: radial inward
            default:
              u = 0;
              v = 0;
          }
          const len = 0.1;
          field.push(x, y, x + u * len, y + v * len);
        }
      }
      drawShape(field, gl.LINES, [0, 1, 0, 1]);

      requestAnimationFrame(render);
    };
    render();
  }, []);

  return (
    <div className="overlay" title="Shows sheath edge and J×B drift using WebGL">
      <h4>Sheath &amp; J×B</h4>
      <canvas ref={canvasRef} width="200" height="200" title="Live WebGL canvas" />
      <div>Phase: {phaseNames[phase]}</div>
      <details>
        <summary>What is this?</summary>
        The cyan disc approximates the sheath boundary and orange lines depict
        J×B drift. Radius follows voltage; arrow length scales with pressure.
        Green arrows overlay a synthetic vector field that changes with the
        annotated phase.
      </details>
    </div>
  );
}
