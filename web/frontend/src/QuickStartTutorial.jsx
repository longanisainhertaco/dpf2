import React, { useState, useEffect } from 'react';

export default function QuickStartTutorial({ setVoltage, setPressure }) {
  const labs = [
    {
      name: 'Setup',
      voltage: 1.0,
      pressure: 0.2,
      description: 'Begin with baseline parameters to initialize the simulation.',
      question: { prompt: 'Which parameter affects particle acceleration?', answer: 'voltage' },
    },
    {
      name: 'Compression',
      voltage: 3.0,
      pressure: 0.1,
      description: 'Ramp up voltage to drive the sheath inward.',
      question: { prompt: 'What force pushes the sheath inward?', answer: 'jxb' },
    },
    {
      name: 'Pinch',
      voltage: 4.5,
      pressure: 0.05,
      description: 'Observe the plasma pinch at peak compression.',
      question: { prompt: 'What forms at the axis during pinch?', answer: 'plasma' },
    },
  ];

  const qs = (window.help && window.help.quickstart) || {};

  const [index, setIndex] = useState(0);
  const [answer, setAnswer] = useState('');
  const [results, setResults] = useState(() => {
    const saved = localStorage.getItem('quickStartTutorialResults');
    return saved ? JSON.parse(saved) : {};
  });

  useEffect(() => {
    const lab = labs[index];
    setVoltage(lab.voltage);
    setPressure(lab.pressure);
    setAnswer(results[lab.name]?.answer || '');
  }, [index]);

  useEffect(() => {
    localStorage.setItem('quickStartTutorialResults', JSON.stringify(results));
  }, [results]);

  const nextStep = () => setIndex((i) => Math.min(i + 1, labs.length - 1));
  const prevStep = () => setIndex((i) => Math.max(i - 1, 0));

  const handleSubmit = () => {
    const lab = labs[index];
    const correct = answer.trim().toLowerCase() === lab.question.answer.toLowerCase();
    const updated = { ...results, [lab.name]: { answer, correct } };
    setResults(updated);
    setAnswer('');
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

  const currentLab = labs[index];

  return (
    <div className="overlay" title={qs.container}>
      <h4 title={qs.header}>Quick Start Tutorial</h4>
      <p title={qs.progress}>
        Step {index + 1} / {labs.length}
      </p>
      <div>
        <button type="button" onClick={prevStep} disabled={index === 0} title={qs.prev}>
          Prev
        </button>
        <button
          type="button"
          onClick={nextStep}
          disabled={index === labs.length - 1}
          title={qs.next}
        >
          Next
        </button>
      </div>
      <p title={qs.description}>{currentLab.description}</p>
      <div className="question" title={qs.questionSection}>
        <p>{currentLab.question.prompt}</p>
        <input
          type="text"
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          title={qs.answerInput}
        />
        <button type="button" onClick={handleSubmit} title={qs.submit}>
          Submit
        </button>
        {results[currentLab.name] && (
          <p title={qs.feedback}>
            {results[currentLab.name].correct ? 'Correct' : 'Incorrect'}
          </p>
        )}
      </div>
      <button type="button" onClick={shareTutorial} title={qs.share}>
        Share Tutorial
      </button>
    </div>
  );
}

