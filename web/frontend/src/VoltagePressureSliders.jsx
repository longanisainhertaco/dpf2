import React from 'react';

// Combined voltage and pressure sliders for simulation control
export default function VoltagePressureSliders({
  voltage,
  pressure,
  setVoltage,
  setPressure,
  onChange,
}) {
  const update = (v, p) => {
    if (onChange) onChange(v, p);
  };

  const handleVoltage = (e) => {
    const v = parseFloat(e.target.value);
    setVoltage(v);
    update(v, pressure);
  };

  const handlePressure = (e) => {
    const p = parseFloat(e.target.value);
    setPressure(p);
    update(voltage, p);
  };

  return (
    <div className="sliders">
      <div>
        <label>
          Voltage: {voltage.toFixed(2)} kV
          <input
            type="range"
            min="0"
            max="5"
            step="0.1"
            value={voltage}
            onChange={handleVoltage}
          />
        </label>
      </div>
      <div>
        <label>
          Pressure: {pressure.toFixed(2)} bar
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={pressure}
            onChange={handlePressure}
          />
        </label>
      </div>
    </div>
  );
}

