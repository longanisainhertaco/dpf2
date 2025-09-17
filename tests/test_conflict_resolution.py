#!/usr/bin/env python3
"""
Test suite for conflict resolution functionality.
"""

import subprocess
import tempfile
import os
from pathlib import Path
import shutil


def test_conflict_detection():
    """Test basic conflict detection functionality."""
    # Create a temporary git repository
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "config", "user.name", "Test User"], cwd=repo_path, check=True
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            check=True,
        )

        # Create initial file
        test_file = repo_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        subprocess.run(["git", "add", "test.py"], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True
        )

        # Rename master to main for consistency
        subprocess.run(["git", "branch", "-m", "main"], cwd=repo_path, check=True)

        # Create a branch and modify the file
        subprocess.run(["git", "checkout", "-b", "feature"], cwd=repo_path, check=True)
        test_file.write_text("line1\nmodified line2\nline3\n")
        subprocess.run(["git", "add", "test.py"], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Feature change"], cwd=repo_path, check=True
        )

        # Go back to main and make conflicting change
        subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True)
        test_file.write_text("line1\ndifferent line2\nline3\n")
        subprocess.run(["git", "add", "test.py"], cwd=repo_path, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Main change"], cwd=repo_path, check=True
        )

        # Copy our conflict resolution script to the test repo
        script_src = Path(__file__).parent.parent / "scripts" / "resolve_conflicts.py"
        script_dst = repo_path / "scripts" / "resolve_conflicts.py"
        script_dst.parent.mkdir(exist_ok=True)
        shutil.copy2(script_src, script_dst)

        # Now try to merge and see if our script detects conflicts
        subprocess.run(["git", "checkout", "feature"], cwd=repo_path, check=True)

        # Attempt merge (should fail with conflicts)
        result = subprocess.run(
            ["git", "merge", "main"], cwd=repo_path, capture_output=True
        )
        assert result.returncode != 0, "Merge should have failed due to conflicts"

        # Run our conflict detection script
        result = subprocess.run(
            ["python", "scripts/resolve_conflicts.py", "--verbose"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        print("Conflict detection output:")
        print(result.stdout)
        print(result.stderr)

        # The script should detect conflicts and exit with error code
        assert (
            result.returncode != 0
        ), "Conflict resolution should have detected conflicts"
        assert (
            "conflicted files" in result.stdout.lower()
            or "conflict" in result.stderr.lower()
        )

        print("✅ Conflict detection test passed!")


def test_hook_installation():
    """Test pre-commit hook installation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True)

        # Copy our conflict resolution script
        script_src = Path(__file__).parent.parent / "scripts" / "resolve_conflicts.py"
        script_dst = repo_path / "scripts" / "resolve_conflicts.py"
        script_dst.parent.mkdir(exist_ok=True)
        shutil.copy2(script_src, script_dst)

        # Install hooks
        result = subprocess.run(
            ["python", "scripts/resolve_conflicts.py", "--install-hooks"],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Hook installation failed: {result.stderr}"

        # Check that the hook was created
        hook_path = repo_path / ".git" / "hooks" / "pre-commit"
        assert hook_path.exists(), "Pre-commit hook was not created"
        assert hook_path.is_file(), "Pre-commit hook is not a file"

        # Check that the hook is executable
        assert os.access(hook_path, os.X_OK), "Pre-commit hook is not executable"

        print("✅ Hook installation test passed!")


if __name__ == "__main__":
    test_conflict_detection()
    test_hook_installation()
    print("🎉 All tests passed!")
