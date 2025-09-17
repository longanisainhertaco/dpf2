# Conflict Resolution Guide for DPF2

This guide provides comprehensive procedures for detecting, resolving, and preventing merge conflicts in the DPF2 repository.

## Quick Start

### Check for Conflicts in Your PR

```bash
# Using our conflict resolution script
python scripts/resolve_conflicts.py --pr <PR_NUMBER>

# Or check locally
git fetch origin
git merge-tree $(git merge-base HEAD origin/main) HEAD origin/main
```

### Auto-Resolve Simple Conflicts

```bash
# Attempt automatic resolution
python scripts/resolve_conflicts.py --auto --strategy smart

# Install conflict prevention hooks
python scripts/resolve_conflicts.py --install-hooks
```

## Types of Conflicts and Resolution Strategies

### 1. Formatting Conflicts

**Common causes:**
- Different code formatting (Python black, C++ clang-format)
- Whitespace differences
- Line ending differences

**Resolution:**
```bash
# Auto-format Python code
python -m black .

# Auto-resolve formatting conflicts
python scripts/resolve_conflicts.py --auto --strategy smart

# For C++ files (if applicable)
clang-format -i src/**/*.cpp src/**/*.h
```

### 2. Import/Include Conflicts

**Common causes:**
- Different import orders
- New imports added in parallel
- Duplicate imports

**Resolution:**
```bash
# Auto-resolve import conflicts
python scripts/resolve_conflicts.py --auto

# Manual resolution for complex cases
# 1. Merge import sections manually
# 2. Remove duplicates
# 3. Sort alphabetically
```

### 3. Configuration File Conflicts

**Common causes:**
- Changes to `pyproject.toml`, `requirements.txt`, or config files
- Version bumps
- Dependency additions

**Resolution:**
```bash
# Manual review required - be careful with dependencies
git checkout --ours pyproject.toml    # Keep our version
git checkout --theirs requirements.txt # Take their version
# Or edit manually to merge both sets of changes
```

### 4. Documentation Conflicts

**Common causes:**
- Parallel updates to README.md, CHANGELOG.md
- Different documentation styles

**Resolution:**
```bash
# Usually requires manual merge
# 1. Review both versions
# 2. Combine information logically
# 3. Maintain consistent style
```

### 5. Code Logic Conflicts

**Common causes:**
- Changes to the same function/method
- Different implementations of similar features
- Refactoring conflicts

**Resolution:**
- **Manual review required**
- Understand both changes
- Write tests to verify correct behavior
- Consider if both changes can coexist

## Resolution Workflows

### Workflow 1: Rebase Strategy (Recommended)

```bash
# 1. Fetch latest changes
git fetch origin

# 2. Switch to your feature branch
git checkout feature-branch

# 3. Rebase onto main
git rebase origin/main

# 4. Resolve conflicts as they appear
# For each conflicted file:
#   - Edit the file to resolve conflicts
#   - git add <file>
#   - git rebase --continue

# 5. Force push (safely)
git push --force-with-lease origin feature-branch
```

### Workflow 2: Merge Strategy

```bash
# 1. Fetch latest changes
git fetch origin

# 2. Switch to your feature branch
git checkout feature-branch

# 3. Merge main into your branch
git merge origin/main

# 4. Resolve conflicts
python scripts/resolve_conflicts.py --auto
# Or resolve manually

# 5. Commit the merge
git commit -m "Merge main into feature-branch"

# 6. Push changes
git push origin feature-branch
```

### Workflow 3: Automated Resolution

```bash
# Use our conflict resolution script
python scripts/resolve_conflicts.py --auto --strategy smart --verbose

# Check what was resolved
git status
git diff --cached

# Commit if satisfied
git commit -m "Resolve merge conflicts"
```

## Conflict Prevention Best Practices

### 1. Regular Rebasing

```bash
# Rebase your branch regularly (daily or before each push)
git fetch origin
git rebase origin/main
```

### 2. Small, Focused Commits

- Keep pull requests small and focused
- Avoid large refactoring in feature branches
- Break large changes into multiple PRs

### 3. Pre-commit Hooks

```bash
# Install our pre-commit hooks
python scripts/resolve_conflicts.py --install-hooks

# Or use pre-commit package
pip install pre-commit
pre-commit install
```

### 4. Code Formatting

```bash
# Format Python code before committing
python -m black .

# Check formatting
python -m black --check .

# Lint code
flake8 .
mypy .
```

### 5. Communication

- Coordinate with other developers on overlapping work
- Use draft PRs for early feedback
- Mention related PRs in discussions

## Advanced Conflict Resolution

### Using Git Merge Tools

```bash
# Configure a merge tool (e.g., VS Code)
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait $MERGED'

# Use the merge tool
git mergetool
```

### Understanding Conflict Markers

```
<<<<<<<< HEAD (your changes)
your_code_here
========
their_code_here
>>>>>>>> branch-name (their changes)
```

### Three-way Merge Resolution

```bash
# See all three versions
git show :1:filename  # common ancestor
git show :2:filename  # your version
git show :3:filename  # their version
```

## Emergency Procedures

### Abort a Merge

```bash
git merge --abort
```

### Abort a Rebase

```bash
git rebase --abort
```

### Reset to Clean State

```bash
# Warning: This will lose uncommitted changes!
git reset --hard HEAD
git clean -fd
```

### Recover Lost Work

```bash
# Find lost commits
git reflog

# Recover a specific commit
git cherry-pick <commit-hash>
```

## Script Usage Examples

### Basic Conflict Detection

```bash
# Check if current branch has conflicts
python scripts/resolve_conflicts.py

# Check specific PR
python scripts/resolve_conflicts.py --pr 636

# Verbose output
python scripts/resolve_conflicts.py --verbose
```

### Automatic Resolution

```bash
# Try to auto-resolve everything
python scripts/resolve_conflicts.py --auto

# Use specific strategy
python scripts/resolve_conflicts.py --auto --strategy ours
python scripts/resolve_conflicts.py --auto --strategy theirs
python scripts/resolve_conflicts.py --auto --strategy smart  # default
```

### Conflict Prevention

```bash
# Install hooks
python scripts/resolve_conflicts.py --install-hooks

# Check repository health
python scripts/resolve_conflicts.py --repo-path /path/to/repo
```

## Integration with CI/CD

The repository includes automated conflict detection in GitHub Actions:

1. **Conflict Detection**: Automatically runs on all PRs
2. **Status Reporting**: Updates PR status with conflict information
3. **Auto-Resolution**: Attempts to resolve simple conflicts automatically
4. **Comments**: Provides detailed guidance for manual resolution

## Troubleshooting

### Common Issues

1. **"fatal: not a git repository"**
   - Run commands from the repository root
   - Check that `.git` directory exists

2. **"conflict resolution failed"**
   - Review conflicts manually
   - Use `git status` to see conflicted files
   - Edit files to remove conflict markers

3. **"cannot force push"**
   - Use `--force-with-lease` instead of `--force`
   - Ensure you have the latest changes: `git fetch origin`

4. **"merge tool not found"**
   - Install and configure a merge tool
   - Use built-in tools: `git config --global merge.tool vimdiff`

### Getting Help

1. Check this documentation
2. Use `python scripts/resolve_conflicts.py --help`
3. Look at GitHub Actions logs for automated resolution attempts
4. Ask for help in PR comments or team channels

## Related Documentation

- [Git Workflow Guide](CONTRIBUTING.md)
- [PIConGPU Commit Rules](pic-dev/docs/COMMIT.md)
- [Continuous Integration](docs/acceptance/ci_dependency_pinning.md)

---

*This guide is maintained by the DPF2 development team. Please update it as new conflict patterns emerge.*