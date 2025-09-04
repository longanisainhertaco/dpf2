import React from 'react';

/**
 * Provides preset experiments for common discharge phases.
 * Selecting a lab adjusts the main voltage and pressure sliders.
 */
export default function GuidedLabs({ setVoltage, setPressure }) {
  const labs = [
    {
      name: 'Breakdown',
      voltage: 1.5,
      pressure: 0.2,
      description:
        'Increase voltage to ionize the gas. Observe initial sheath growth.',
    },
    {
      name: 'Rundown',
      voltage: 3.0,
      pressure: 0.1,
      description:
        'Current drives the sheath toward the axis; track J×B direction.',
    },
    {
      name: 'Pinch',
      voltage: 4.5,
      pressure: 0.05,
      description:
        'Sheath collapses and compresses plasma into a dense pinch.',
    },
  ];

  return (
    <div className="overlay" title="Step-by-step exploration of discharge phases">
      <h4>Guided Labs</h4>
      {labs.map((lab) => (
        <div key={lab.name}>
          <button
            type="button"
            onClick={() => {
              setVoltage(lab.voltage);
              setPressure(lab.pressure);
            }}
            title={`Configure parameters for the ${lab.name} phase`}
          >
            {lab.name}
          </button>
          <p>{lab.description}</p>
        </div>
      ))}
      <details>
        <summary>What is this?</summary>
        Each lab loads representative parameters for a given phase. Use it to
        rapidly explore how voltage and pressure influence the simulation.
      </details>
    </div>
  );
}
