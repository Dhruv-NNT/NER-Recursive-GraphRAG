# Running Docker from a User-Space Tarball (No sudo)

Some HPC environments block package installs or `sudo`. You can still run Docker in **rootless** mode by unpacking the official Docker tarball inside your home directory.

## 1. Download and unpack the static Docker binaries

```bash
cd $HOME
wget https://download.docker.com/linux/static/stable/x86_64/docker-26.1.3.tgz
tar -xzf docker-26.1.3.tgz
mv docker docker-static
export PATH="$HOME/docker-static:$PATH"
```

Add the `PATH` export to your shell profile (`~/.bashrc`) so future shells can find the binaries.

## 2. Enable rootless mode

Docker ships with a helper script for rootless setups. Run it once to check prerequisites and configure your `$HOME` instance:

```bash
$HOME/docker-static/dockerd-rootless-setuptool.sh check
$HOME/docker-static/dockerd-rootless-setuptool.sh install
```

The script:
1. Creates the necessary runtime directories (`$XDG_RUNTIME_DIR/docker`).
2. Prints systemd/user service commands (if available) or a fallback command to start `dockerd-rootless.sh` manually.

If the HPC cluster does not offer systemd user services, start the daemon manually inside a screen/tmux session:

```bash
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
export PATH="$HOME/docker-static:$PATH"
$HOME/docker-static/dockerd-rootless.sh --experimental
```

Keep this process running. All Docker CLI commands in other shells must include the same `DOCKER_HOST` and `PATH` environment variables (add them to `~/.bashrc`).

## 3. Use Docker as usual

Once the rootless daemon is running, the normal Docker CLI works:

```bash
docker version
docker info
docker run hello-world
```

If the CLI cannot connect, double-check that:

- `$DOCKER_HOST` points to the socket path printed by the setup tool.
- `$XDG_RUNTIME_DIR` is exported (common default: `/run/user/$(id -u)`).
- The daemon is running in another terminal (no `dockerd-rootless` process means no socket).

## 4. Stopping / restarting

- Manual daemon: terminate the `dockerd-rootless.sh` process (Ctrl+C) and restart using the same command.
- Systemd user service: `systemctl --user stop docker` / `systemctl --user start docker`.

## 5. Clean-up

To remove the rootless environment, delete the `$HOME/.local/share/docker` directory and stop the daemon. The original system-wide Docker installation (if any) remains untouched.

> **Tip:** Rootless Docker cannot expose privileged low ports (<1024). Map Neo4j/Ollama ports above 1024 when using this setup.
