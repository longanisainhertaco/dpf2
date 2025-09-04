import React from 'react';

/**
 * Drag-and-drop selection for common DPF geometries.
 * Dropping a preset notifies the parent via onSelect.
 */
export default function GeometryPresets({ onSelect }) {
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

  const handleDragStart = (e, value) => {
    e.dataTransfer.setData('text/plain', value);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const val = e.dataTransfer.getData('text/plain');
    if (val && onSelect) onSelect(val);
  };

  const handleDragOver = (e) => e.preventDefault();

  return (
    <div className="overlay" title="Choose a preset electrode geometry">
      <h4>Geometry Presets</h4>
      <div className="presets">
        {presets.map((p) => (
          <div
            key={p.value}
            draggable
            onDragStart={(e) => handleDragStart(e, p.value)}
            className="preset"
            title={p.description}
          >
            {p.name}
          </div>
        ))}
      </div>
      <div
        className="dropzone"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        title="Drag a preset here to select it"
      >
        Drop Here
      </div>
      <details>
        <summary>What is this?</summary>
        Drag either the Mather or Filippov option into the drop zone to apply
        that geometry to a new configuration set.
      </details>
    </div>
  );
}

