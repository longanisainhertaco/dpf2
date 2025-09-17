#!/usr/bin/env python3
"""
Conflict Resolution Tool for DPF2 Repository

This script helps detect and resolve merge conflicts in pull requests,
providing automated conflict resolution for common cases and guidance for manual resolution.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConflictResolver:
    """Handles detection and resolution of merge conflicts in pull requests."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.conflicts_detected = []

    def run_git_command(
        self, command: List[str], capture_output: bool = True
    ) -> subprocess.CompletedProcess:
        """Execute a git command and return the result."""
        full_command = ["git"] + command
        logger.debug(f"Running: {' '.join(full_command)}")

        result = subprocess.run(
            full_command, cwd=self.repo_path, capture_output=capture_output, text=True
        )

        if result.returncode != 0 and capture_output:
            logger.error(f"Git command failed: {' '.join(full_command)}")
            logger.error(f"Error: {result.stderr}")

        return result

    def check_pr_mergeable(self, pr_number: int) -> Tuple[bool, str]:
        """Check if a PR is mergeable using GitHub CLI or API."""
        try:
            # Try to use GitHub CLI if available
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "view",
                    str(pr_number),
                    "--json",
                    "mergeable,mergeableState",
                ],
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                mergeable = data.get("mergeable", False)
                state = data.get("mergeableState", "unknown")
                return mergeable, state

        except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
            logger.warning(
                "GitHub CLI not available or failed, falling back to git checks"
            )

        return self._check_mergeable_locally()

    def _check_mergeable_locally(self) -> Tuple[bool, str]:
        """Check mergeable status using local git operations."""
        # Get current branch
        current_branch = self.run_git_command(
            ["branch", "--show-current"]
        ).stdout.strip()

        # Fetch latest changes
        self.run_git_command(["fetch", "origin"])

        # Try a test merge
        result = self.run_git_command(["merge-tree", f"origin/main", "HEAD"])

        if result.returncode == 0 and not result.stdout.strip():
            return True, "clean"
        elif "<<<<<<< " in result.stdout:
            return False, "dirty"
        else:
            return True, "behind"

    def detect_conflict_types(
        self, conflicted_files: List[str]
    ) -> Dict[str, List[str]]:
        """Categorize conflicts by type for targeted resolution strategies."""
        conflict_types = {
            "formatting": [],
            "imports": [],
            "version_conflicts": [],
            "documentation": [],
            "configuration": [],
            "code_logic": [],
        }

        for file_path in conflicted_files:
            if any(
                file_path.endswith(ext) for ext in [".py", ".js", ".ts", ".cpp", ".h"]
            ):
                if "test" in file_path.lower():
                    conflict_types["code_logic"].append(file_path)
                elif any(
                    keyword in file_path.lower()
                    for keyword in ["format", "style", "lint"]
                ):
                    conflict_types["formatting"].append(file_path)
                elif "import" in file_path.lower() or file_path.endswith("__init__.py"):
                    conflict_types["imports"].append(file_path)
                else:
                    conflict_types["code_logic"].append(file_path)
            elif file_path.endswith((".md", ".rst", ".txt")):
                conflict_types["documentation"].append(file_path)
            elif any(
                file_path.endswith(ext)
                for ext in [".json", ".yml", ".yaml", ".toml", ".cfg"]
            ):
                conflict_types["configuration"].append(file_path)
            elif "version" in file_path.lower() or "changelog" in file_path.lower():
                conflict_types["version_conflicts"].append(file_path)
            else:
                conflict_types["code_logic"].append(file_path)

        return {k: v for k, v in conflict_types.items() if v}

    def auto_resolve_formatting_conflicts(self, files: List[str]) -> bool:
        """Automatically resolve formatting-related conflicts."""
        logger.info("Attempting to auto-resolve formatting conflicts...")

        for file_path in files:
            try:
                # For Python files, try using black formatter
                if file_path.endswith(".py"):
                    result = subprocess.run(
                        ["python", "-m", "black", file_path],
                        cwd=self.repo_path,
                        capture_output=True,
                    )
                    if result.returncode == 0:
                        logger.info(f"Auto-formatted {file_path} with black")

                # For other files, try removing conflict markers and using the "ours" version
                self._resolve_with_strategy(file_path, "ours")

            except Exception as e:
                logger.warning(f"Failed to auto-resolve {file_path}: {e}")
                return False

        return True

    def auto_resolve_import_conflicts(self, files: List[str]) -> bool:
        """Automatically resolve import-related conflicts."""
        logger.info("Attempting to auto-resolve import conflicts...")

        for file_path in files:
            try:
                with open(self.repo_path / file_path, "r") as f:
                    content = f.read()

                # Remove duplicate imports and sort
                resolved_content = self._merge_imports(content)

                with open(self.repo_path / file_path, "w") as f:
                    f.write(resolved_content)

                logger.info(f"Auto-resolved imports in {file_path}")

            except Exception as e:
                logger.warning(f"Failed to auto-resolve imports in {file_path}: {e}")
                return False

        return True

    def _merge_imports(self, content: str) -> str:
        """Merge conflicting import statements."""
        lines = content.split("\n")
        resolved_lines = []
        in_conflict = False
        ours_imports = []
        theirs_imports = []

        for line in lines:
            if line.startswith("<<<<<<< "):
                in_conflict = True
                continue
            elif line.startswith("======="):
                continue
            elif line.startswith(">>>>>>> "):
                # Merge the imports
                all_imports = list(set(ours_imports + theirs_imports))
                resolved_lines.extend(sorted(all_imports))
                in_conflict = False
                ours_imports = []
                theirs_imports = []
                continue

            if in_conflict:
                if line.strip().startswith(("import ", "from ")):
                    if (
                        "======="
                        not in content[
                            content.find("<<<<<<< ") : content.find(">>>>>>> ")
                        ]
                    ):
                        ours_imports.append(line)
                    else:
                        theirs_imports.append(line)
            else:
                resolved_lines.append(line)

        return "\n".join(resolved_lines)

    def _resolve_with_strategy(self, file_path: str, strategy: str = "ours"):
        """Resolve conflicts using a specific strategy (ours, theirs, or manual)."""
        if strategy == "ours":
            self.run_git_command(["checkout", "--ours", file_path])
        elif strategy == "theirs":
            self.run_git_command(["checkout", "--theirs", file_path])

    def generate_conflict_report(self, conflicts: Dict[str, List[str]]) -> str:
        """Generate a detailed conflict resolution report."""
        report = ["# Conflict Resolution Report", ""]

        total_files = sum(len(files) for files in conflicts.values())
        report.append(f"Total conflicted files: {total_files}")
        report.append("")

        for conflict_type, files in conflicts.items():
            if files:
                report.append(
                    f"## {conflict_type.title()} Conflicts ({len(files)} files)"
                )
                for file_path in files:
                    report.append(f"- {file_path}")
                report.append("")

                # Add resolution suggestions
                if conflict_type == "formatting":
                    report.append(
                        "**Suggested resolution:** Run formatting tools (black, clang-format)"
                    )
                elif conflict_type == "imports":
                    report.append(
                        "**Suggested resolution:** Merge imports and remove duplicates"
                    )
                elif conflict_type == "documentation":
                    report.append(
                        "**Suggested resolution:** Manual review and merge of documentation"
                    )
                elif conflict_type == "configuration":
                    report.append(
                        "**Suggested resolution:** Careful manual review of configuration changes"
                    )
                else:
                    report.append(
                        "**Suggested resolution:** Manual code review and testing"
                    )

                report.append("")

        return "\n".join(report)

    def resolve_conflicts(
        self, auto_resolve: bool = True, strategy: str = "smart"
    ) -> bool:
        """Main conflict resolution workflow."""
        logger.info("Starting conflict resolution process...")

        # Check if we're in a merge state
        merge_head_path = self.repo_path / ".git" / "MERGE_HEAD"
        if not merge_head_path.exists():
            logger.info("No active merge found. Checking for potential conflicts...")
            mergeable, state = self._check_mergeable_locally()
            if mergeable:
                logger.info("No conflicts detected.")
                return True
            else:
                logger.warning(f"Potential conflicts detected (state: {state})")

        # Get list of conflicted files
        result = self.run_git_command(["diff", "--name-only", "--diff-filter=U"])
        conflicted_files = [f for f in result.stdout.strip().split("\n") if f]

        if not conflicted_files:
            logger.info("No conflicted files found.")
            return True

        logger.info(f"Found {len(conflicted_files)} conflicted files:")
        for file_path in conflicted_files:
            logger.info(f"  - {file_path}")

        # Categorize conflicts
        conflicts = self.detect_conflict_types(conflicted_files)

        # Generate report
        report = self.generate_conflict_report(conflicts)
        print("\n" + report)

        if auto_resolve:
            resolved_count = 0

            # Auto-resolve formatting conflicts
            if conflicts.get("formatting"):
                if self.auto_resolve_formatting_conflicts(conflicts["formatting"]):
                    resolved_count += len(conflicts["formatting"])

            # Auto-resolve import conflicts
            if conflicts.get("imports"):
                if self.auto_resolve_import_conflicts(conflicts["imports"]):
                    resolved_count += len(conflicts["imports"])

            logger.info(f"Auto-resolved {resolved_count} files")

            # Stage resolved files
            if resolved_count > 0:
                self.run_git_command(["add"] + conflicted_files[:resolved_count])
        else:
            resolved_count = 0

        return len(conflicted_files) - resolved_count == 0

    def create_conflict_prevention_hooks(self):
        """Create pre-commit hooks to prevent common conflicts."""
        hooks_dir = self.repo_path / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)

        pre_commit_hook = hooks_dir / "pre-commit"

        hook_content = """#!/bin/bash
# DPF2 Conflict Prevention Pre-commit Hook

# Run formatting checks
echo "Running formatting checks..."
if command -v python3 &> /dev/null && python3 -m black --check . 2>/dev/null; then
    echo "✓ Python formatting OK"
else
    echo "⚠ Python formatting issues detected. Run 'python -m black .' to fix."
fi

# Check for potential merge conflicts in commit
if git diff --cached --check; then
    echo "✓ No whitespace errors"
else
    echo "⚠ Whitespace errors detected"
    exit 1
fi

# Check for leftover conflict markers
if git diff --cached | grep -E "^\\+.*(<<<<<<<|=======|>>>>>>>)"; then
    echo "❌ Conflict markers found in staged changes!"
    exit 1
fi

echo "✓ Pre-commit checks passed"
"""

        with open(pre_commit_hook, "w") as f:
            f.write(hook_content)

        # Make executable
        os.chmod(pre_commit_hook, 0o755)
        logger.info("Created pre-commit hook for conflict prevention")


def main():
    parser = argparse.ArgumentParser(
        description="Resolve merge conflicts in DPF2 repository"
    )
    parser.add_argument("--pr", type=int, help="Pull request number to check")
    parser.add_argument(
        "--auto", action="store_true", help="Attempt automatic resolution"
    )
    parser.add_argument(
        "--strategy",
        choices=["ours", "theirs", "smart"],
        default="smart",
        help="Conflict resolution strategy",
    )
    parser.add_argument(
        "--install-hooks",
        action="store_true",
        help="Install pre-commit hooks for conflict prevention",
    )
    parser.add_argument("--repo-path", default=".", help="Path to repository")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    resolver = ConflictResolver(args.repo_path)

    if args.install_hooks:
        resolver.create_conflict_prevention_hooks()
        print("Conflict prevention hooks installed.")
        return

    if args.pr:
        mergeable, state = resolver.check_pr_mergeable(args.pr)
        print(f"PR #{args.pr} - Mergeable: {mergeable}, State: {state}")

        if not mergeable:
            print("PR has conflicts that need resolution.")

    success = resolver.resolve_conflicts(auto_resolve=args.auto, strategy=args.strategy)

    if success:
        print("✅ All conflicts resolved successfully!")
        sys.exit(0)
    else:
        print("❌ Some conflicts require manual resolution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
