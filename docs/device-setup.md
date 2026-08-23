# Device setup

Bringing up a machine with a Hailo-10H: PCIe driver, HailoRT, and
hailo-ollama. The compile side (Docker toolchain) is covered in
[../docker/README.md](../docker/README.md); this page covers the device
host.

> **Distribution channel.** Hailo packages its software in two ways:
> the **Hailo Software Suite** download from the
> [Developer Zone](https://hailo.ai/developer-zone/) (free account, .tar
> package with all components), and the **meta-hailo** Yocto layers for
> embedded builds. Both carry the same components at matching versions.
> Keep driver / HailoRT / server versions aligned — mismatches surface as
> cryptic PCIe or firmware errors.

## Components and where they come from

| Component | License | Source |
|---|---|---|
| `hailo_pci` kernel driver | GPL-2.0 | Software Suite (`hailort-drivers`) or [meta-hailo](https://github.com/hailo-ai/meta-hailo) |
| HailoRT (libraries + `hailortcli`) | MIT (core) | Software Suite, or build from [hailo-ai/hailort](https://github.com/hailo-ai/hailort) (public source) |
| Firmware (flashed on device) | proprietary | shipped with the suite; loaded by the driver |
| genai LLM runtime (`hailo_platform.genai`, C++ server) | proprietary | ships inside the suite's HailoRT package for LLM-capable parts |
| hailo-ollama | proprietary | ships with the software suite for the Hailo-10H; no public repository |

Version pinning used to validate everything in this repo:

- DFC **5.3.0** (the LLM flow — prefill/tbt + KV-cache — must be present;
  it appeared in the 5.x line with `set_kv_cache_global_params`)
- matching HailoRT 5.x from the same suite drop

## Install order

1. **PCIe driver.** From the suite:
   ```bash
   cd hailort-drivers/linux/pci_driver && make all && insmod hailo_pci.ko
   ```
   Verify: `lspci | grep Hailo` and `dmesg | tail` showing firmware load.
2. **Firmware check.** The Hailo-10H needs an LLM-capable firmware build;
   `hailortcli fw-control identify` prints the part number/firmware version.
3. **HailoRT user space.** Install the `.deb`/`.rpm` packages from the
   suite (libhailort + hailortcli + python bindings). Verify:
   ```bash
   hailortcli scan          # lists devices
   hailortcli fw-control identify
   ```
4. **genai Python module.** Part of the suite's HailoRT python package
   (`from hailo_platform.genai import LLM`). Verify with a one-liner import.
5. **hailo-ollama.** Install from the suite; then:
   ```bash
   OLLAMA_HOST=0.0.0.0:8000 hailo-ollama serve &
   curl -s http://localhost:8000/api/tags     # lists registered models
   ```

## Memory limits

The DFC optimization step can request large amounts of RAM, and loading
multi-GB HEFs into RAM on small hosts will OOM them. Practical rules:

- Cap the compile container: `docker run --memory=24g ...` on a 32 GB host.
- On small devices (single-board computers with 4 GB RAM), never slurp a
  HEF into memory wholesale — use bounded seek+read access per resource,
  as [../runtime/diagnostics/hef_audit.py](../runtime/diagnostics/hef_audit.py)
  does.
- If the device drops into a degraded power state, recover it with a full
  power cycle of the host — repeated software probing of a degraded device
  can wedge the whole system.

## Sanity checks after install

```bash
# Device visible and healthy?
hailortcli scan && hailortcli fw-control identify

# Python stack sees it?
python - <<'PY'
import hailo_platform as hpf
print(hpf.VDevice().get_part_number_string())
PY
```

Then compile something ([../pipeline/README.md](../pipeline/README.md)),
copy the HEF over, register it
([../runtime/register_hailo_ollama.py](../runtime/register_hailo_ollama.py))
and generate.
