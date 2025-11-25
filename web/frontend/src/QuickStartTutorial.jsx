import React, { useMemo } from 'react';

/**
 * A guided flow that links the visible overlays to the underlying physics
 * levers. Each step sets recommended values and describes what to look for in
 * the multi-pane plots and WebGL sheath viewer.
 */
export default function QuickStartTutorial({ setVoltage, setPressure }) {
  const steps = useMemo(
    () => [
      {
        title: 'Charge and Breakdown',
        action: () => {
          setVoltage(1.8);
          setPressure(0.08);
        },
        description:
          'Increase the bank voltage while keeping neutral pressure modest. The voltage pane should show a fast ramp while the sheath plot begins wide—mirroring early breakdown physics and the regime gauge should sit just under the magnetised threshold.',
      },
      {
        title: 'Rundown and Compression',
        action: () => {
          setVoltage(2.4);
          setPressure(0.12);
        },
        description:
          'Raise pressure to thicken the sheath and slow the current surge. Watch the current pane peak later while the WebGL sheath narrows, illustrating axial rundown and a climb in the dimensionless gate.',
      },
      {
        title: 'Pinch Optimization',
        action: () => {
          setVoltage(3.0);
          setPressure(0.16);
        },
        description:
          'Push into the pinch regime: the sheath radius contracts and the sheath pane spikes. The Regime Dashboard should flip toward magnetized, collisionless values as the gate crosses the dashed line.',
      },
      {
        title: 'Afterglow Diagnostics',
        action: () => {
          setVoltage(1.2);
          setPressure(0.06);
        },
        description:
          'Lower both controls to observe afterglow decay. Note how the current tail aligns with the sheath panel and how regime indicators drift back toward fluid-valid ranges with the gate falling below unity.',
      },
    ],
    [setPressure, setVoltage]
  );

  return (
    <div className="overlay" title="Walk through the basic DPF simulation controls">
      <h4>Tutorial Flow</h4>
      <p>
        Each checkpoint syncs the sliders with the multi-pane plots, tying the
        on-screen visuals to the dominant plasma physics process. Click a step
        to replay the state and compare overlays.
      </p>
      <ol>
        {steps.map((step) => (
          <li key={step.title}>
            <div className="tutorial-step">
              <div>
                <strong>{step.title}.</strong> {step.description}
              </div>
              <button type="button" onClick={step.action}>
                Load step
              </button>
            </div>
          </li>
        ))}
      </ol>
      <details>
        <summary>What should I look for?</summary>
        Use the current/voltage/sheath panes to see how electrical drive couples
        into sheath dynamics, then confirm the operating regime with the live
        dimensionless dashboard. The gate mixes Lundquist and ωτ so students can
        see when numerics should switch to higher-order, magnetised settings.
        Saving a snapshot preserves these pairings for reproducibility.
      </details>
    </div>
  );
}
