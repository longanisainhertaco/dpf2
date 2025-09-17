"""Tests for the :mod:`dpf2.hpc` job manager."""

from pathlib import Path

from dpf2.hpc import JobManager


def test_slurm_submit_options(tmp_path, monkeypatch):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    jm = JobManager("slurm")

    called: dict[str, object] = {}

    def fake_run(cmd, capture_output, text, check, env):  # type: ignore[override]
        called["cmd"] = cmd
        called["env"] = env

        class R:
            pass

        return R()

    monkeypatch.setattr("subprocess.run", fake_run)

    jm.submit(
        str(script),
        nodes=2,
        nodelist="node1,node2",
        gpus=2,
        gpu_affinity=[0, 1],
        dependencies=[11, 22],
        restart="chkpt.dat",
        script_args=["--foo", "bar"],
    )

    cmd = called["cmd"]
    env = called["env"]
    assert cmd[0] == "sbatch"
    assert "-N" in cmd and cmd[cmd.index("-N") + 1] == "2"
    assert "--nodelist" in cmd and cmd[cmd.index("--nodelist") + 1] == "node1,node2"
    assert "--gpus" in cmd and cmd[cmd.index("--gpus") + 1] == "2"
    assert (
        "--dependency" in cmd and cmd[cmd.index("--dependency") + 1] == "afterok:11:22"
    )
    # The job script is wrapped for staging so we only verify the tail
    # contains the expected script arguments.
    assert cmd[-4:] == ["--foo", "bar", "--restart", "chkpt.dat"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["DPF_RESTART"] == "chkpt.dat"


def test_mpi_node_topology_and_restart(tmp_path, monkeypatch):
    script = tmp_path / "run.py"
    script.write_text("#!/bin/bash\n")
    jm = JobManager("mpi")

    called: dict[str, object] = {}

    def fake_run(cmd, capture_output, text, check, env):  # type: ignore[override]
        called["cmd"] = cmd
        called["env"] = env

        class R:
            pass

        return R()

    monkeypatch.setattr("subprocess.run", fake_run)

    topo = {"hostA": [0, 1], "hostB": [0]}
    jm.submit(str(script), node_topology=topo, restart="chk.dat", gpu_affinity=[0, 1])

    cmd = called["cmd"]
    env = called["env"]
    assert cmd[0] == "mpirun"
    assert "--hostfile" in cmd
    hostfile = Path(cmd[cmd.index("--hostfile") + 1])
    hosts = set(hostfile.read_text().strip().splitlines())
    assert "hostA slots=2" in hosts and "hostB slots=1" in hosts

    gpu_map = Path(env["DPF_GPU_MAP"]).read_text().strip().splitlines()
    assert "0 hostA 0" in gpu_map
    assert "1 hostA 1" in gpu_map
    assert "2 hostB 0" in gpu_map

    # Wrapper script obscures the actual job path; ensure args are forwarded.
    assert cmd[-2:] == ["--restart", "chk.dat"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["DPF_RESTART"] == "chk.dat"


def test_stage_manifest_and_restart(tmp_path, monkeypatch):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    jm = JobManager("slurm")

    called: dict[str, object] = {}

    def fake_wrap(self, job_script, stage_in, stage_out):  # type: ignore[override]
        called["stage_in"] = stage_in
        called["stage_out"] = stage_out
        return job_script

    def fake_run(cmd, capture_output, text, check, env):  # type: ignore[override]
        called["cmd"] = cmd
        called["env"] = env

        class R:
            pass

        return R()

    monkeypatch.setattr(JobManager, "_wrap_staging", fake_wrap)
    monkeypatch.setattr("subprocess.run", fake_run)

    manifest = "run/run_manifest.json"
    jm.submit(str(script), manifest=manifest, restart=manifest)

    assert called["stage_out"][manifest] == manifest
    assert called["stage_in"][manifest] == manifest
    cmd = called["cmd"]
    idx = cmd.index(str(script))
    assert cmd[idx + 1 : idx + 3] == ["--restart", manifest]
    env = called["env"]
    assert env["DPF_RESTART"] == manifest


def test_default_manifest_staged(tmp_path, monkeypatch):
    """Ensure ``run_manifest.json`` is copied even without explicit argument."""

    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    jm = JobManager("slurm")

    called: dict[str, object] = {}

    def fake_wrap(self, job_script, stage_in, stage_out):  # type: ignore[override]
        called["stage_out"] = stage_out
        return job_script

    def fake_run(cmd, capture_output, text, check, env):  # type: ignore[override]
        return type("R", (), {})()

    monkeypatch.setattr(JobManager, "_wrap_staging", fake_wrap)
    monkeypatch.setattr("subprocess.run", fake_run)

    jm.submit(str(script))
    assert called["stage_out"]["run_manifest.json"] == "run_manifest.json"


def test_hdf5_manifest_written(tmp_path, monkeypatch):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\n")
    jm = JobManager("slurm")

    monkeypatch.setattr(JobManager, "_wrap_staging", lambda self, js, si, so: js)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: type("R", (), {})())

    manifest_h5 = tmp_path / "run_manifest.h5"
    cfg = {"x": 1}

    import h5py_stub as h5py, json

    jm.submit(
        str(script),
        manifest_h5=str(manifest_h5),
        config=cfg,
        container_hash="abc123",
    )

    with h5py.File(manifest_h5, "r") as h5:
        grp = h5["manifest"]
        assert grp.attrs["container_hash"] == "abc123"
        assert json.loads(grp.attrs["config"]) == cfg
        assert "git_commit" in grp.attrs
