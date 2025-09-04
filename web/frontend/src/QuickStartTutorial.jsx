
import React from 'react';

/**
 * Basic quick-start instructions for running the simulation.
 * Provides contextual guidance for adjusting voltage and pressure.
 */
export default function QuickStartTutorial({ setVoltage, setPressure }) {
  return (
    <div className="overlay" title="Walk through the basic DPF simulation controls">
      <h4>Quick Start Tutorial</h4>
      <p>
        Submit a configuration to begin exploring the simulation. Adjust the
        voltage and pressure sliders to see how the discharge responds.
      </p>
    </div>
  );
}

