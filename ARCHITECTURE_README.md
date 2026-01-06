# Architecture Analysis - Quick Start

## Viewing the Mermaid Diagram

The data flow sequence diagram is available in `architecture_diagram.mmd`.

### Option 1: GitHub (Recommended)
GitHub automatically renders `.mmd` files. Simply view [architecture_diagram.mmd](architecture_diagram.mmd) in the GitHub web interface.

### Option 2: VS Code
Install the "Markdown Preview Mermaid Support" or "Mermaid Preview" extension, then open the `.mmd` file.

### Option 3: Online Editor
Copy the contents of `architecture_diagram.mmd` and paste into:
- https://mermaid.live/
- https://mermaid-js.github.io/mermaid-live-editor/

### Option 4: Markdown with Mermaid
The full diagram is also embedded in `ARCHITECTURE_ANALYSIS.md` within a markdown code block.

## Quick Summary

The diagram visualizes the flow: **User → Service A (FastAPI Backend) → Database (JSON Files)**

### Major Issues Highlighted in Diagram:
- 🔴 **Critical**: Authentication bypass (token = username)
- 🔴 **Critical**: Results endpoint returns config, not results
- 🔴 **Critical**: Missing authentication on snapshot retrieval
- 🔴 **Critical**: Insecure file upload with no validation
- 🟡 **High**: No actual HPC dispatch (mock implementation)
- 🟡 **Medium**: Predictable timestamp-based IDs

## Full Documentation

- **ARCHITECTURE_ANALYSIS.md** - Complete analysis with 10 documented logic errors
- **evaluation_report.md** - Updated project evaluation with security section
- **architecture_diagram.mmd** - Mermaid sequence diagram source

## Key Findings

1. **No Real Database**: The system uses JSON files on the filesystem, not a database
2. **Broken Authentication**: OAuth2 implementation returns username as token without JWT
3. **Hardcoded Credentials**: Plain-text passwords in source code
4. **Data Integrity**: Results endpoint returns configuration instead of actual results
5. **Missing Implementation**: HPC dispatch is a placeholder that only saves config files
