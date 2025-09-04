# Web App Tutorial

This walkthrough demonstrates how to use the experimental web interface.

## 1. Start the server

```bash
uvicorn web.backend.main:app --reload
```

Navigate to `http://localhost:8000` in your browser.

## 2. Log in

Use the sample credentials `admin/secret` or `user/secret`.
Tooltips on each field describe their purpose.

## 3. Submit a configuration

Paste a JSON configuration into the text area.  The `What is this?` dropdown
explains what the configuration represents.

## 4. Import or export setups

Use the **Export Snapshot** button to download the current setup as
`snapshot.json`.  Choose a JSON file with the file input to import a shared
setup.

## 5. Explore overlays

After submitting, voltage and pressure sliders become active.  The page shows
real‑time overlays:

* **Instability Growth** – visualizes phase‑dependent motion.
* **Sheath & Beam** – illustrates sheath position, J×B drift, and beam formation.

Each overlay contains a `What is this?` section describing the plot.
