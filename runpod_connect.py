import time
import logging
import textwrap
from pathlib import Path

import paramiko
import runpod

from config import runpod_config, model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class RunPodSession:
    """Manages a RunPod pod lifecycle: create → SSH → execute → teardown."""

    def __init__(self):
        runpod.api_key = runpod_config.api_key
        self.pod_id: str | None = None
        self.ssh: paramiko.SSHClient | None = None

    # ------------------------------------------------------------------ #
    # Pod lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def create_pod(self) -> str:
        log.info("Creating RunPod pod...")
        pod = runpod.create_pod(
            name=runpod_config.pod_name,
            image_name=runpod_config.image_name,
            gpu_type_id=runpod_config.gpu_type_id,
            gpu_count=runpod_config.gpu_count,
            container_disk_in_gb=runpod_config.container_disk_in_gb,
            volume_in_gb=runpod_config.volume_in_gb,
            ports="22/tcp,8000/http",
            support_public_ip=True,
            env={
                "HF_TOKEN": model_config.hf_token,
                "PYTHONUNBUFFERED": "1",
            },
        )
        self.pod_id = pod["id"]
        log.info("Pod created: %s", self.pod_id)
        return self.pod_id

    def wait_for_ready(self) -> dict:
        log.info("Waiting for pod to become ready (timeout=%ds)...", runpod_config.pod_ready_timeout)
        deadline = time.time() + runpod_config.pod_ready_timeout
        while time.time() < deadline:
            pod = runpod.get_pod(self.pod_id)
            status = pod.get("desiredStatus") or pod.get("runtime", {})
            if pod.get("runtime") and pod["runtime"].get("ports"):
                log.info("Pod is ready.")
                return pod
            log.info("  status=%s — retrying in 15s...", status)
            time.sleep(15)
        raise TimeoutError(f"Pod {self.pod_id} did not become ready in time.")

    def terminate_pod(self):
        if self.pod_id:
            runpod.terminate_pod(self.pod_id)
            log.info("Pod %s terminated.", self.pod_id)
            self.pod_id = None

    # ------------------------------------------------------------------ #
    # SSH                                                                   #
    # ------------------------------------------------------------------ #

    def _get_ssh_info(self, pod: dict) -> tuple[str, int]:
        """Extract host and SSH port from pod runtime info."""
        for port_entry in pod["runtime"]["ports"]:
            if port_entry.get("privatePort") == 22 or port_entry.get("type") == "tcp":
                host = port_entry["ip"]
                port = int(port_entry["publicPort"])
                return host, port
        raise RuntimeError("SSH port not found in pod runtime info.")

    def connect_ssh(self, pod: dict, ssh_key_path: str = "~/.ssh/id_rsa") -> paramiko.SSHClient:
        host, port = self._get_ssh_info(pod)
        log.info("Connecting via SSH to %s:%d ...", host, port)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        key_path = Path(ssh_key_path).expanduser()
        pkey = paramiko.RSAKey.from_private_key_file(str(key_path))

        deadline = time.time() + runpod_config.ssh_timeout
        while time.time() < deadline:
            try:
                client.connect(hostname=host, port=port, username="root", pkey=pkey, timeout=30)
                log.info("SSH connection established.")
                self.ssh = client
                return client
            except Exception as exc:
                log.warning("SSH not ready yet (%s), retrying in 10s...", exc)
                time.sleep(10)

        raise TimeoutError("Could not establish SSH connection within timeout.")

    # ------------------------------------------------------------------ #
    # Remote execution                                                      #
    # ------------------------------------------------------------------ #

    def exec(self, command: str, stream_output: bool = True) -> tuple[str, str, int]:
        """Run a shell command on the remote pod."""
        if not self.ssh:
            raise RuntimeError("SSH not connected. Call connect_ssh() first.")

        _, stdout, stderr = self.ssh.exec_command(command, get_pty=True)
        out_lines, err_lines = [], []

        for line in iter(stdout.readline, ""):
            if stream_output:
                print(line, end="")
            out_lines.append(line)

        exit_code = stdout.channel.recv_exit_status()
        err_data = stderr.read().decode()
        if err_data:
            err_lines.append(err_data)

        return "".join(out_lines), "".join(err_lines), exit_code

    def exec_script(self, script: str, stream_output: bool = True) -> int:
        """Upload and run an inline Python script on the pod."""
        remote_path = "/tmp/_quant_script.py"
        sftp = self.ssh.open_sftp()
        with sftp.file(remote_path, "w") as f:
            f.write(textwrap.dedent(script))
        sftp.close()
        _, _, exit_code = self.exec(f"python {remote_path}", stream_output=stream_output)
        return exit_code

    def install_dependencies(self):
        log.info("Installing dependencies on pod...")
        cmds = [
            "pip install -q --upgrade pip",
            "pip install -q vllm autoawq gptqmodel bitsandbytes llmcompressor",
            "pip install -q 'aqlm[gpu]'",
            "pip install -q git+https://github.com/SqueezeAILab/SqueezeLLM.git",
            "pip install -q datasets tqdm huggingface_hub",
        ]
        for cmd in cmds:
            log.info("  running: %s", cmd)
            _, _, code = self.exec(cmd, stream_output=False)
            if code != 0:
                log.warning("Command returned non-zero exit code: %s", cmd)

    # ------------------------------------------------------------------ #
    # Context manager                                                       #
    # ------------------------------------------------------------------ #

    def __enter__(self):
        self.create_pod()
        pod = self.wait_for_ready()
        self.connect_ssh(pod)
        self.install_dependencies()
        return self

    def __exit__(self, *_):
        if self.ssh:
            self.ssh.close()
        self.terminate_pod()
