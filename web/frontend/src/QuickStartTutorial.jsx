import React from 'react';
import help from './help.json';

export default function QuickStartTutorial({ setVoltage, setPressure }) {
  const applyExample = () => {
    setVoltage(2.5);
    setPressure(0.15);
  };

  return (
    <div className="overlay" title={help.tutorial.root}>
      <h4>Quick Start Tutorial</h4>
      <p>Follow these steps to run a simple simulation.</p>
      <button type="button" onClick={applyExample} title={help.tutorial.apply}>
        Apply Example Settings
      </button>
    </div>
  );
}
