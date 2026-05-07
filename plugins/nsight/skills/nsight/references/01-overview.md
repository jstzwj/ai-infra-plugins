# NVIDIA Nsight Systems -- Overview and Installation Reference

## 1. What Is Nsight Systems

NVIDIA Nsight Systems is a statistical sampling profiler with tracing features. It is designed to work with devices and devkits based on NVIDIA Tegra SoCs (system-on-chip), Arm SBSA (server based system architecture) systems, and systems based on the x86_64 processor architecture that also include NVIDIA GPU(s).

Throughout this documentation the following terminology is used:

- **Target** -- The device on which profiling happens.
- **Host** -- The computer on which the user works and controls the profiling session.

For x86_64 based systems the target and host may be on the same device. For Tegra or Arm based systems they will always be separate.

### 1.1 Tracing vs Sampling

Nsight Systems distinguishes three different activities:

| Activity | Description |
|----------|-------------|
| **Profiling** | The process of collecting any performance data. A profiling session in Nsight Systems typically includes sampling and tracing. |
| **Sampling** | The process of periodically stopping the profilee (the application under investigation), typically to collect backtraces (call stacks of active threads), which allows you to understand statistically how much time is spent in each function. Additionally, hardware counters can also be sampled. This process is inherently imprecise when a low number of samples have been collected. |
| **Tracing** | The process of collecting precise information about various activities happening in the profilee or in the system. For example, profilee API execution may be traced providing the exact time and duration of a function call. |

### 1.2 Common Features

Common features supported by Nsight Systems on most platforms include:

- Sampling of the profilee and collecting backtraces using multiple algorithms (frame pointers, DWARF data, Intel Last Branch Record). Building top-down, bottom-up, and flat views.
- Sampling or tracing system power behaviors, such as CPU frequency.
- (Embedded Platforms Edition only) Sampling counters from Arm PMU (Performance Monitoring Unit).
- Support for multiple windows and multiple monitors.

With Nsight Systems a user can:

- Identify call paths that monopolize the CPU.
- Identify individual functions that monopolize the CPU across different call paths.
- (Embedded Platforms Edition) Identify functions with poor cache utilization.
- See CUDA Runtime and Driver API calls, and CUDA GPU workload via CUPTI.
- See NVTX annotations: ranges, markers, and thread names.
- (Windows) See D3D12 API calls, graphic frames, stutter analysis, and GPU workloads.
- (x86_64) See Vulkan API calls, graphic frames, stutter analysis, and GPU workloads.

### 1.3 Editions

| Edition | Target Platforms |
|---------|-----------------|
| Nsight Systems Embedded Platforms Edition | NVIDIA Tegra products for the embedded and automotive market (Linux for Tegra, QNX) |
| Nsight Systems Workstation Edition | x86_64 and Arm server (SBSA) processors for workstation, cluster, and cloud (Linux, Windows) |

---

## 2. Supported Platforms

### 2.1 L4T (Linux for Tegra)

Based on your Jetson version, select the appropriate JetPack:

- Current Jetson targets: [NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- Older Tegra targets: [NVIDIA JetPack Archives](https://developer.nvidia.com/embedded/jetpack-archive)

### 2.2 x86_64 or Arm SBSA

- **GPU Architectures:** Starting with Turing (Pascal and Volta were dropped in version 2025.2; Power PC was dropped in version 2024.2)
- **Operating Systems (64 bit only):**
  - Ubuntu 20.04, 22.04, and 24.04
  - CentOS 8.0 and RedHat Enterprise Linux 8+
  - Amazon Linux 2023+
  - Windows Server 2022+
- **Networking Components:**
  - NVIDIA DPUs
  - NVIDIA SuperNICs
  - Amazon EFA NICs

### 2.3 CUDA Version Compatibility

Nsight Systems supports CUDA 10.0+ for most platforms. Nsight Systems on Arm SBSA supports 10.2+.

Note that CUDA version and driver version must be compatible.

| CUDA Version | Minimum Driver Version |
|--------------|----------------------|
| 11.0 | 450 |
| 10.2 | 440.30 |
| 10.1 | 418.39 |
| 10.0 | 410.48 |

From CUDA 11.X on, any driver from 450 on will be supported, although new features introduced in more recent drivers will not be available.

---

## 3. System Requirements

### 3.1 Requirements for x86_64 and Arm SBSA Targets on Linux

When attaching to x86_64 or Arm SBSA Linux-based target from the GUI on the host, the connection is established through SSH.

**Use of Linux Perf:** To collect thread scheduling data and IP (instruction pointer) samples, the Linux operating system's `perf_event_paranoid` level must be 2 or less.

Check the current level:

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

If the output is greater than 2, temporarily adjust:

```bash
sudo sh -c 'echo 2 >/proc/sys/kernel/perf_event_paranoid'
```

To make the change permanent:

```bash
sudo sh -c 'echo kernel.perf_event_paranoid=2 > /etc/sysctl.d/local.conf'
```

**Kernel version requirements:**
- 3.10.0-693 or later for CentOS and RedHat Enterprise Linux 7.4+
- 4.3 or greater for all other distros including Ubuntu

Check kernel version:

```bash
uname -a
```

**glibc version:** Nsight Systems requires glibc 2.17 or newer.

```bash
ldd --version
```

**CUDA:** Use the `deviceQuery` command to determine the CUDA driver and runtime versions on the system. It is normally installed at:

```
/usr/local/cuda/samples/1_Utilities/deviceQuery
```

**Additional requirements:**
- Only pure 64-bit environments are supported.
- Nsight Systems requires write permission to the `/var/lock` directory on the target system.
- Docker: See Container Support section for more information.

### 3.2 Requirements for x86_64 Targets on Windows

DX12 Requires:
- Windows 10 with NVIDIA Driver 411.63 or higher for DX12 trace
- Windows 10 April 2018 Update (version 1803, AKA Redstone 4) with NVIDIA Driver 411.63 or higher for DirectX Ray Tracing and DX12 Copy command queues

### 3.3 Requirements for QNX Targets

**Development environment:**
- Nsight Systems supports profiling DRIVE OS QNX targets in development environments.
- Some features require additional setup (see Profiling Embedded VMs).

**Safety environment:**
- Nsight Systems provides limited profiling capabilities in QNX Safety environment.
- The `prod_debug_extra` overlay is required to enable Nsight Systems in safety environment.

> **Warning:** Nsight Systems is a profiling and analysis tool that is not safety-certified. It must not be used in environments where software controls driving decisions or impacts human safety.

Available features in QNX Safety:

| Feature | First Supported In |
|---------|-------------------|
| Tracelogger trace (CPU thread states and context switches) | 6.0.8.x |
| Hypervisor trace (VM context switches, interrupts, traps, etc.) | 6.0.8.x |
| VMProfiler (Cross-Hypervisor sampling) | 6.0.8.x |
| OSRT trace (trace of C runtime functions) | 6.0.8.x |
| NVTX trace (trace of user-added NVTX instrumentation) | 6.0.8.x |

### 3.4 Host Application Requirements

The Nsight Systems host application runs on the following host platforms:
- Windows 10, Windows Server 2019 (64-bit only)
- Linux Ubuntu 14.04 and higher (64-bit only)
- macOS 10.10 "Yosemite" and higher (arm64 supported in 2025.2+)

---

## 4. Installation Methods

### 4.1 Finding the Right Package

Choose the right package based on target system:

- **Tegra targets:** Nsight Systems Embedded Platforms Edition (part of NVIDIA JetPack SDK)
- **x86_64 or Arm SBSA:** Nsight Systems Workstation Edition from [developer.nvidia.com/nsight-systems](https://developer.nvidia.com/nsight-systems)
- The x86_64 and Arm SBSA target versions are also available in the CUDA Toolkit

Each package is limited to one architecture.

**Tegra packages:**
- Windows host: Install `.msi` -- remote access to Tegra device
- Linux host: Install `.run` -- remote access to Tegra device
- macOS host: Install `.dmg` -- remote access to Tegra device

**x86_64 packages:**
- Windows host: Install `.msi` -- remote access to Linux x86_64 or Windows devices; also local
- Linux host: Install `.run`, `.rpm`, or `.deb` -- remote or localhost
- Linux CLI only: Install `.deb` or `.rpm` -- CLI collection only
- macOS host: Install `.dmg` -- remote access to Linux x86_64 device

**Arm SBSA packages:**
- Arm SBSA host: Install `.run`, `.rpm`, or `.deb` -- local profiling and report viewing
- Arm SBSA CLI only: Install `.deb` or `.rpm` -- CLI collection only

### 4.2 Package Manager Installation

#### Ubuntu (minimal setup for containers)

Assumes root in the container. Example command to launch a container: `sudo docker run -it --rm ubuntu:latest bash`

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

#### Ubuntu (desktop)

```bash
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
sudo apt install nsight-systems
```

#### CentOS and RHEL (minimal setup for containers)

```bash
rpm --import https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
dnf install -y 'dnf-command(config-manager)'
dnf config-manager --add-repo "https://developer.download.nvidia.com/devtools/repos/rhel$(source /etc/os-release; echo ${VERSION_ID%%.*})/$(rpm --eval '%{_arch}' | sed s/aarch/arm/)/"
dnf install -y nsight-systems-cli
```

#### CentOS and RHEL (desktop)

```bash
sudo rpm --import https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dnf install -y 'dnf-command(config-manager)'
sudo dnf config-manager --add-repo "https://developer.download.nvidia.com/devtools/repos/rhel$(source /etc/os-release; echo ${VERSION_ID%%.*})/$(rpm --eval '%{_arch}' | sed s/aarch/arm/)/"
sudo dnf install nsight-systems
```

### 4.3 Manual Installation

Copy the appropriate file to your host system in a directory where you have write and execute permissions. Run the install file, accept the EULA, and Nsight Systems will install.

On Linux, automated installation options:
- `--accept` flag: automatically accept the EULA
- `--accept --quiet`: accept EULA without printing to stdout
- Running with `--quiet` without `--accept` will display an error

The installation creates a `Host` directory for this host and a `Target` directory for each supported target. All binaries needed on the target will be installed by the host on first connection.

If installing from the CUDA Toolkit, see the CUDA Toolkit documentation.

---

## 5. CLI Setup and PATH Configuration

All Nsight Systems targets can be profiled using the CLI. The CLI is especially helpful when scripts are used to run unattended collections or when access to the target system via SSH is not possible.

The CLI can be found in the `Target` directory of the Nsight Systems installation. Users who want to install the CLI as a standalone tool can copy the files within the `Target` directory to the location of their choice.

If you wish to run the CLI without root (recommended mode), install in a directory where you have full access.

### 5.1 Environment Check

Once you have the CLI set up, use the `nsys status -e` command to check your environment:

```bash
~$ nsys status -e

Sampling Environment Check
Linux Kernel Paranoid Level = 1: OK
Linux Distribution = Ubuntu
Linux Kernel Version = 4.15.0-109-generic: OK
Linux perf_event_open syscall available: OK
Sampling trigger event available: OK
Intel(c) Last Branch Record support: Available
Sampling Environment: OK
```

This status check allows you to ensure that the system requirements for CPU sampling are met. If the Sampling Environment is not OK, you will still be able to run various trace operations.

Intel(c) Last Branch Record allows tools to use hardware to quickly get limited stack information. Nsight Systems will use this method for stack resolution by default if available.

To get started using the CLI, run `nsys --help` for a list of options.

---

## 6. GUI Launching

Depending on your OS, Nsight Systems will have installed an icon on your host desktop that you can use to launch the GUI. To launch the GUI directly, run the `nsys-ui` executable in the `Host` sub-directory of your installation.

### 6.1 Remote Profiling from the GUI

Nsight Systems provides a simple interface to profile on localhost or manage multiple connections to Linux or Windows based devices via SSH. The network connections manager can be launched through the device selection dropdown.

> **Security notice:** SSH is only used to establish the initial connection to a target device, perform checks, and upload necessary files. The actual profiling commands and data are transferred through a raw, unencrypted socket. Nsight Systems should not be used in a network setup where attacker-in-the-middle attack is possible, or where untrusted parties may have network access to the target device.

While connecting to the target device, you will be prompted to input the user's password. If you choose to remember the password, it will be stored in plain text in the configuration file on the host. Stored passwords are bound to the public key fingerprint of the remote device.

**No authentication option** is useful for devices configured for passwordless login using root username. Edit `/etc/ssh/sshd_config` on the target:

```ini
PermitRootLogin yes
```

Then set empty password using `passwd` and restart SSH: `service ssh restart`.

### 6.2 Open Ports

The Nsight Systems daemon requires port 22 and port 45555 to be open for listening.

```bash
# Check open ports
sudo firewall-cmd --list-ports --permanent
sudo firewall-cmd --reload

# Open a port permanently
sudo firewall-cmd --permanent --add-port 45555/tcp
sudo firewall-cmd --reload
```

On cloud systems, you must open port 22 and port 45555 for ingress.

### 6.3 Kernel Version Number on Target

```bash
cat /proc/quadd/version
# Minimal supported version is 1.82
```

Netcat (nc) is required on the target device:

```bash
sudo apt-get install netcat-openbsd
```

### 6.4 Hotkey Trace Start/Stop

Nsight Systems Workstation Edition can use hotkeys to control profiling. Press the hotkey to start and/or stop a trace session from within the target application's graphic window. The default hotkey is F12.

On Windows, a different hotkey binding can be configured by setting `HotKeyIntValue` in `config.ini`:

```ini
HotKeyIntValue=112
```

The value is the decimal numeric identifier of the virtual key. For example, 112 corresponds to F1.

### 6.5 Symbol Locations (Windows)

Symbol resolution happens on host, and therefore does not affect performance of profiling on the target. Use the Symbol locations dialog to specify:
- Paths of PDB files
- Symbol servers
- Location of the local symbol cache

---

## 7. Jupyter Notebook Integration

### 7.1 Installing Multi Report Analysis System

> **PREVIEW FEATURE**

The Nsight Systems multi-report analysis system can be located in the `<install-dir>/target-linux-x64/python/packages` directory. For this initial release, multi-node analysis is only available to run recipes on Linux targets, and only available to visualize on Linux or Windows hosts.

**Recipe Dependencies:**
- Python 3.8 or newer with pip and venv

Pip/venv on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install python3-pip
sudo apt-get install python3-venv
```

#### Automated Installation Script

The `<install-dir>/target-linux-x64/python/packages/nsys_recipe/install.py` script automates dependency installation.

Options:
- `-h`: Display help
- `--current`: Install packages in the current environment
- `--venv PATH`: Install packages in a virtual environment (created if needed)
- `--tar`: Download wheel packages online and tar them
- `--untar`: Untar the wheel packages and install
- `--python`: Change the python executable (default is `python3`)
- `--no-jupyter`: Do not install requirements for the Jupyter notebook
- `--no-dask`: Do not install requirements for Dask

#### Manual Installation

Create a virtual environment:

```bash
python3 -m venv recipe_env
source recipe_env/bin/activate
```

Dependency files are located in `<install-dir>/target-linux-x64/python/packages/nsys_recipe/requirements/`:
- `Common.txt` (required)
- `Dask.txt` (optional)
- `Jupyter.txt` (optional)

One-step installation:

```bash
python3 -m pip install -r nsys_recipe/requirements/dask.txt \
  -r nsys_recipe/requirements/common.txt \
  -r nsys_recipe/requirements/jupyter.txt
```

Two-step installation (for machines without internet):

On the machine with internet:
```bash
python3 -m pip download -r nsys_recipe/requirements/dask.txt \
  -r nsys_recipe/requirements/common.txt \
  -r nsys_recipe/requirements/jupyter.txt -d "recipe-deps"
tar -cvfz recipe-deps.tar.gz recipe-deps
```

On the machine without internet:
```bash
tar -xvfz recipe-deps.tar.gz
python3 -m pip install recipe-deps/* --no-index
```

### 7.2 Jupyter Notebook Configuration

The Nsight Systems UI can internally load a Jupyter notebook. It uses the Jupyter installation associated with the Python on your `$PATH`.

If Jupyter is installed elsewhere, add a variable to `config.ini`:

```ini
JupyterPythonExe="/path/to/recipe_env/bin/python"
```

Place `config.ini` in `<install_dir>/host-linux-x64`.

On Windows:
```ini
JupyterPythonExe="c:\\path\\to\\recipe_env\\bin\\python.exe"
```

### 7.3 Profiling within JupyterLab

The JupyterLab Nsight extension integrates Nsight Systems profiling into JupyterLab for profiling of Jupyter notebook cells. CUDA kernels launched by the cells as well as CUDA and Python code execution can be profiled and analyzed.

For more information and to install the extension: JupyterLab Nsight extension on PyPI.

---

## 8. Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **Profiling** | The process of collecting performance data from an application. In Nsight Systems, a profiling session typically includes sampling and tracing. |
| **Sampling** | Periodically stopping the profilee to collect backtraces (call stacks of active threads). This provides a statistical view of where time is spent. Inherently imprecise at low sample counts. |
| **Tracing** | Collecting precise information about activities in the profilee or system. For example, API execution tracing provides exact time and duration of function calls. |
| **NVTX** | NVIDIA Tools Extension. A user-instrumentation API that allows developers to annotate their code with ranges, markers, and thread names. Nsight Systems can visualize these annotations on the timeline. |
| **Report** | The output file (`.nsys-rep`) generated by Nsight Systems after a profiling session. Contains all collected data and can be opened in the GUI or exported to various formats. |
| **Backtrace** | A call stack captured at a specific point in time. Backtraces are collected during sampling to show which functions were active and how they were called. |
| **CUPTI** | CUDA Profiling Tools Interface. The underlying API used by Nsight Systems to collect CUDA trace data. |
| **IP Sample** | Instruction Pointer sample. A snapshot of the CPU instruction pointer location, collected periodically during sampling. |
| **Context Switch** | An event where the OS scheduler moves a thread off or onto a CPU core. Tracing context switches shows thread scheduling behavior. |
| **Process Tree** | The launched process and all of its descendant processes. Nsight Systems can trace the entire process tree. |
| **Session** | A sequence of CLI commands that define one or more collections. A session begins with `start`, `launch`, or `profile` and ends with `shutdown` or when a profile command terminates. Multiple sessions can run concurrently on the same system. |
| **Capture Range** | A mechanism to control when profiling data is collected within a running application, using NVTX ranges, CUDA profiler API, or hotkeys. |
| **Metric Set** | A predefined collection of GPU hardware counters/metrics that can be sampled during profiling. |
| **SQLite Export** | Conversion of `.nsys-rep` data into an SQLite database for programmatic analysis via the `nsys stats` or `nsys export` commands. |

---

## 9. Architecture and Data Flow

### 9.1 Text-Based Architecture Diagram

```
+-------------------+        +-------------------+        +-------------------+
|                   |        |                   |        |                   |
|   User / Script   |        |    Nsight Systems |        |     Target App    |
|                   |        |       CLI         |        |                   |
+--------+----------+        +--------+----------+        +--------+----------+
         |                            |                            |
         | nsys profile <app>         | Launch & inject            |
         |----------------------------|---------------------------->|
         |                            |                            |
         |                            |    +------------------+    |
         |                            |    |   Profiling      |    |
         |                            |    |   Collectors:    |    |
         |                            |    |  - CPU Sampling  |    |
         |                            |    |  - API Trace     |    |
         |                            |    |  - GPU Metrics   |    |
         |                            |    |  - NVTX          |    |
         |                            |    +------------------+    |
         |                            |                            |
         |                            |<--- .nsys-rep file --------|
         |                            |                            |
+--------+----------+        +--------+----------+                 |
|                   |        |                   |                 |
|   Nsight Systems |        |    nsys stats     |                 |
|       GUI        |<-------|    nsys export    |                 |
|                   |  open  |                   |                 |
+-------------------+  .nsys-rep                |
         |                                +-------------------+
         |                                |                   |
         |  View timeline,                |  External tools   |
         |  analyze data                  |  (SQLite, CSV,    |
         |                                |   JSON, HDF5)     |
         +--------------------------------|                   |
                                          +-------------------+
```

### 9.2 Data Flow Description

1. **Collection Phase:** The CLI (`nsys`) launches the target application with appropriate injection libraries. Collectors (CPU sampling, API tracing, GPU metrics, NVTX) gather data during execution.

2. **Output Phase:** When collection completes, Nsight Systems writes all collected data into a `.nsys-rep` file. Optionally, additional export files (SQLite, HDF5, JSON, CSV, Arrow, Parquet) can be generated.

3. **Analysis Phase:** The `.nsys-rep` file can be:
   - Opened in the Nsight Systems GUI for interactive timeline visualization
   - Processed with `nsys stats` for statistical summaries
   - Processed with `nsys analyze` for expert system analysis
   - Exported with `nsys export` for use with external tools

4. **Remote Profiling:** When profiling remote targets, the GUI connects via SSH, deploys target binaries, controls collection, and transfers the `.nsys-rep` file back to the host.

### 9.3 Report File Types

| Extension | Format | Description |
|-----------|--------|-------------|
| `.nsys-rep` | Binary | Primary Nsight Systems report file. Opened in GUI or processed by CLI. |
| `.sqlite` | SQLite DB | Exported relational database. Used by `nsys stats` and `nsys analyze`. |
| `.h5` | HDF5 | Hierarchical Data Format export. Supported only on x86_64 Linux and Windows. |
| `.txt` | Text | Human-readable text export. |
| `.json` | JSON | JSON export. |
| `.arrows` / `_arwdir` | Apache Arrow | Arrow format export. |
| `_pqtdir` | Apache Parquet | Parquet directory export. |

---

## 10. Container and Scheduler Support

### 10.1 Collecting Data Within a Container

While examples in this section use Docker container semantics, other containers work much the same. Nsight Systems strongly recommends using the CLI to profile in a container. Best container practice is to split services across containers when they do not require colocation.

**Enable Docker Collection:**

There are three ways to enable the `perf_event_open` system call:
1. `--privileged=true` switch
2. `--cap-add=SYS_ADMIN` switch
3. Seccomp security profile (requires seccomp 2.2.1+)

To use the seccomp approach, download the default profile, remove the `perf_event_open` line if guarded by `CAP_SYS_ADMIN`, and add the following under "syscalls":

```json
{
  "name": "perf_event_open",
  "action": "SCMP_ACT_ALLOW",
  "args": []
}
```

Save as `default_with_perf.json` and start Docker with:

```
--security-opt seccomp=default_with_perf.json
```

Example Docker launch:

```bash
sudo nvidia-docker run --network=host \
  --security-opt seccomp=default_with_perf.json \
  --rm -ti caffe-demo2 bash
```

After the Docker has been started, use the Nsight Systems CLI to launch a collection within the Docker. The resulting file can be imported into the Nsight Systems host like any other CLI result.

### 10.2 Profiling Services Launched via Kubernetes

Nsight Systems provides profiling via sidecar injection without modifying containers or k8s/helm specs. Data can be filtered by namespace or pod using Kubernetes labels, or by command-line regex.

Compatible with AKS, EKS, GKE, and OKE. Documentation and download at [NGC Nsight Operator](https://ngc.nvidia.com).

### 10.3 Nsight Streamer for Nsight Systems

A self-hosted NVIDIA Nsight Systems GUI running inside a Docker container enables remote access through a web browser. Available at [NGC](https://ngc.nvidia.com).

### 10.4 GUI VNC Container

Nsight Systems provides a build script to build a self-isolated Docker container with the GUI and VNC server. Located at `host-linux-x64/Scripts/VncContainer/build.py`. Requires Docker and Python 3.5+.

Build Parameters:

| Short Name | Full Name | Description |
|------------|-----------|-------------|
| --vnc-password | (optional) | Default VNC password (at least 6 characters) |
| -aba | --additional-build-arguments | Additional Docker build arguments |
| -hd | --nsys-host-directory | Directory with Nsight Systems host binaries |
| -td | --nsys-target-directory | Directory with Nsight Systems target binaries (repeatable) |
| --tigervnc | (optional) | Use TigerVNC instead of x11vnc |
| --http | (optional) | Install noVNC for HTTP access |
| --rdp | (optional) | Install xRDP for RDP access |
| --geometry | (optional) | Default VNC resolution WidthxHeight (default 1920x1080) |
| --build-directory | (optional) | Directory for temporary files |

Ports:

| Port | Purpose | Condition |
|------|---------|-----------|
| TCP 5900 | VNC access | Always |
| TCP 80 | HTTP access to noVNC | Built with `--http` |
| TCP 3389 | RDP access | Built with `--rdp` |

Volumes:

| Docker Folder | Purpose |
|---------------|---------|
| `/mnt/host` | Root path for shared folders |
| `/mnt/host/Projects` | Folder with projects and reports |
| `/mnt/host/logs` | Folder with inner services logs |

Environment Variables:

| Variable | Purpose |
|----------|---------|
| `VNC_PASSWORD` | Password for VNC access (at least 6 characters) |
| `NSYS_WINDOW_WIDTH` | Width of VNC display in pixels |
| `NSYS_WINDOW_HEIGHT` | Height of VNC display in pixels |

Example usage:

```bash
# VNC on port 5916
sudo docker run -p 5916:5900/tcp -ti nsys-ui-vnc:1.0

# VNC + HTTP
sudo docker run -p 5916:5900/tcp -p 8080:80/tcp -ti nsys-ui-vnc:1.0

# VNC + HTTP + RDP
sudo docker run -p 5916:5900/tcp -p 8080:80/tcp -p 33890:3389/tcp -ti nsys-ui-vnc:1.0

# With shared home, custom resolution, and password
sudo docker run -p 5916:5900/tcp \
  -v $HOME:/mnt/host/home \
  -e NSYS_WINDOW_WIDTH=3840 -e NSYS_WINDOW_HEIGHT=2160 \
  -e VNC_PASSWORD=7654321 \
  -ti nsys-ui-vnc:1.0

# With shared home and projects folder
sudo docker run -p 5916:5900/tcp \
  -v $HOME:/mnt/host/home \
  -v /opt/NsysProjects:/mnt/host/Projects \
  -ti nsys-ui-vnc:1.0
```

---

## 11. Known Issues and Important Notes

### 11.1 Architecture Deprecations

- **Nsight Systems 2025.2+** does not support Pascal or Volta architectures. Use an older version from [developer.nvidia.com/gameworksdownload](https://developer.nvidia.com/gameworksdownload).
- **Nsight Systems 2024.2+** does not support Power PC. Use an older version.
- **Nsight Systems 2024.4+** does not support cuBLAS versions prior to 11.4.

### 11.2 General Limitations

- Session names are limited to 127 characters; executable names are limited to 111 characters for `nsys profile`.
- Profiling greater than 5 minutes is not officially supported. Start with short sessions.
- Attaching or re-attaching to a process from the GUI is not supported with x86_64 Linux target. Use the interactive CLI instead.
- By default, Nsight Systems traces a subset of API calls likely to impact performance, not all calls.
- There is an upper bound on the default size for recording trace events. If hit, reduce profiling duration or number of features traced.
- Using Nsight Systems with applications that use CUPTI (like some TensorFlow versions) may not work due to CUPTI limitations.
- Tracing applications with non-thread-safe memory allocators is not supported.
- Tracing OS Runtime libraries in an application that preloads glibc symbols is unsupported and can lead to undefined behavior.
- Nsight Systems cannot profile applications launched through a virtual window manager like GNU Screen.
- Using Nsight Systems MPI trace functionality with the Darshan runtime module can lead to segfaults. Unload with `module unload darshan-runtime`.

### 11.3 WSL Timestamp Note

The default time conversion is not reliable on WSL. Set `CuptiUseRawGpuTimestamps` to false:

```bash
mkdir -p "$(dirname "$(nsys -z)")"
echo 'CuptiUseRawGpuTimestamps=false' >> "$(nsys -z)"
```

### 11.4 vGPU Considerations

- Always use the profiler grant when running on vGPU.
- Starting with vGPU 13.0, device level metrics collection is exposed to end users.
- Nsight Systems is supported in vGPU environments requiring a vGPU license (CUDA 11.4+, R470 TRD1 driver+).
- If the license is not obtained after 20 minutes, GPU performance metrics data will be inaccurate.

### 11.5 Docker Issues

- In a Docker, when a system's host utilizes a kernel older than v4.3, it is not possible to collect sampling data unless both the host and Docker are running RHEL or CentOS with kernel 3.10.1-693+.
- When `docker exec` is called on a running container and stdout is kept open, the exec shell hangs until the command exits. Use `docker exec --tty` to avoid.

### 11.6 CUDA Trace Issues

- If a system is in CC-DevTools mode and Nsight Systems traces CUDA in an application using libcrypto, Nsight Systems may crash when the application exits. Workarounds include: adding `cudaDeviceSynchronize` before exit, adding `cudaProfilerStop` with `--flush-on-cudaprofilerstop=true`, setting a collection duration, or using capture ranges.
- CUDA GPU trace collection requires a fraction of GPU memory. If the application utilizes all GPU memory, trace might not work.
- On Tegra platforms, CUDA trace requires root privileges.
- For applications using multiple streams from multiple threads, CUDA event buffers may not be released properly.

### 11.7 Python Multiprocessing

The Python multiprocessing module defaults to "fork" mode on Linux, which leads to undefined behavior for tools that rely on injection. Use `set_start_method("spawn")` for safer profiling:

```python
import multiprocessing as mp
mp.set_start_method("spawn")
```

Ensure processes exit gracefully using `close` and `join` methods. Otherwise, Nsight Systems cannot flush buffers properly and traces may be missing.
