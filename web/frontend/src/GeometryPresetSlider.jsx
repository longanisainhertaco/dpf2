import React, { useState, useEffect } from 'react';

// Slider based selector for common DPF geometries
export default function GeometryPresetSlider({ onSelect }) {
  const presets = [
    {
      name: 'Mather',
      value: 'mather',
      description:
        'Classic coaxial Mather-type geometry suited for high-density operation.',
    },
    {
      name: 'Filippov',
      value: 'filippov',
      description:
        'Filippov geometry with a broad anode offering a short rundown time.',
    },
  ];

  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (onSelect) onSelect(presets[index].value);
  }, [index]);

  return (
    <div className="overlay" title="Slide to choose a preset geometry">
      <h4>Geometry Presets</h4>
      <input
        type="range"
        min="0"
        max={presets.length - 1}
        step="1"
        value={index}
        onChange={(e) => setIndex(parseInt(e.target.value))}
      />
      <p>{presets[index].name}</p>
      <details>
        <summary>What is this?</summary>
        Drag the slider to toggle between common electrode geometries. The
        currently selected preset is reported back to the parent component.
      </details>
    </div>
  );
}

