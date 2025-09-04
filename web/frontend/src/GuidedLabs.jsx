import React, { useState, useEffect } from 'react';

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

  const [currentLab, setCurrentLab] = useState(null);
  const [userAnswer, setUserAnswer] = useState('');
  const [results, setResults] = useState(() => {
    const saved = localStorage.getItem('guidedLabResults');
    return saved ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    localStorage.setItem('guidedLabResults', JSON.stringify(results));
  }, [results]);

  const score = Object.values(results).filter((r) => r.correct).length;

  const handleSubmit = () => {
    if (!currentLab) return;
    const correct =
      userAnswer.trim().toLowerCase() ===
      currentLab.question.answer.toLowerCase();
    const updated = {
      ...results,
      [currentLab.name]: { answer: userAnswer, correct },
    };
    setResults(updated);
    setUserAnswer('');
    setCurrentLab(null);
  };

  return (
    <div className="overlay" title="Step-by-step exploration of discharge phases">
      <h4>Guided Labs</h4>
      <p>
        Score: {score} / {labs.length}
      </p>
      {labs.map((lab) => (
        <div key={lab.name}>
          <button
            type="button"
            onClick={() => {
              setVoltage(lab.voltage);
              setPressure(lab.pressure);
              setCurrentLab(lab);
            }}
            title={`Configure parameters for the ${lab.name} phase`}
          >
            {lab.name}
          </button>
          <p>{lab.description}</p>
          {results[lab.name] && (
            <p>
              {results[lab.name].correct
                ? 'Answered correctly'
                : 'Answered incorrectly'}
            </p>
          )}
        </div>
      ))}
      {currentLab && (
        <div className="question">
          <h5>{currentLab.name} Question</h5>
          <p>{currentLab.question.prompt}</p>
          <input
            type="text"
            value={userAnswer}
            onChange={(e) => setUserAnswer(e.target.value)}
          />
          <button type="button" onClick={handleSubmit}>
            Submit
          </button>
        </div>
      )}
      <details>
        <summary>What is this?</summary>
        Each lab loads representative parameters for a given phase. Use it to
        rapidly explore how voltage and pressure influence the simulation.
      </details>
    </div>
  );
}
