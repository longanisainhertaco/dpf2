import React, { useState, useEffect } from 'react';
import help from './help.json';

/**
 * Step through key discharge phases with preset parameters.
 * Each step adjusts the main voltage and pressure sliders.
 */
export default function QuickStartTutorial({ setVoltage, setPressure }) {
  const labs = [
    {
      name: 'Breakdown',
      voltage: 1.5,
      pressure: 0.2,
      description:
        'Increase voltage to ionize the gas. Observe initial sheath growth.',
      question: {
        prompt: 'What causes gas breakdown?',
        answer: 'ionization',
      },
    },
    {
      name: 'Rundown',
      voltage: 3.0,
      pressure: 0.1,
      description:
        'Current drives the sheath toward the axis; track J×B direction.',
      question: {
        prompt: 'Which force pushes the sheath inward?',
        answer: 'jxb',
      },
    },
    {
      name: 'Pinch',
      voltage: 4.5,
      pressure: 0.05,
      description:
        'Sheath collapses and compresses plasma into a dense pinch.',
      question: {
        prompt: 'What forms at the axis during pinch?',
        answer: 'plasma',
      },
    },
  ];

  const [step, setStep] = useState(0);
  const [userAnswer, setUserAnswer] = useState('');
  const [results, setResults] = useState(() => {
    const saved = localStorage.getItem('quickStartResults');
    return saved ? JSON.parse(saved) : {};
  });

  const currentLab = labs[step];

  useEffect(() => {
    localStorage.setItem('quickStartResults', JSON.stringify(results));
  }, [results]);

  useEffect(() => {
    setVoltage(currentLab.voltage);
    setPressure(currentLab.pressure);
  }, [step, currentLab, setVoltage, setPressure]);

  const handleSubmit = () => {
    const correct =
      userAnswer.trim().toLowerCase() ===
      currentLab.question.answer.toLowerCase();
    const updated = {
      ...results,
      [currentLab.name]: { answer: userAnswer, correct },
    };
    setResults(updated);
    setUserAnswer('');
  };

  const shareTutorial = () => {
    const bundle = { labs, results };
    const blob = new Blob([JSON.stringify(bundle, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'tutorial_bundle.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="overlay" title={help.quickstart.container}>
      <h4>Quick Start</h4>
      <p>
        Step {step + 1} / {labs.length}
      </p>
      <p>{currentLab.description}</p>
      {results[currentLab.name] && (
        <p>
          {results[currentLab.name].correct
            ? 'Answered correctly'
            : 'Answered incorrectly'}
        </p>
      )}
      <div className="question">
        <h5>{currentLab.name} Question</h5>
        <p>{currentLab.question.prompt}</p>
        <input
          type="text"
          value={userAnswer}
          onChange={(e) => setUserAnswer(e.target.value)}
        />
        <button
          type="button"
          onClick={handleSubmit}
          title={help.quickstart.submit}
        >
          Submit
        </button>
      </div>
      <div>
        <button
          type="button"
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          title={help.quickstart.prev}
        >
          Prev
        </button>
        <button
          type="button"
          onClick={() => setStep((s) => Math.min(labs.length - 1, s + 1))}
          disabled={step === labs.length - 1}
          title={help.quickstart.next}
        >
          Next
        </button>
        <button
          type="button"
          onClick={shareTutorial}
          title={help.quickstart.share}
        >
          Share Tutorial
        </button>
      </div>
      <details>
        <summary>What is this?</summary>
        Each lab loads representative parameters for a given phase. Use it to
        rapidly explore how voltage and pressure influence the simulation.
      </details>
    </div>
  );
}
