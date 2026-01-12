import sounddevice as sd

def main():
    print("=== sounddevice device list ===")
    try:
        devices = sd.query_devices()
    except Exception as e:
        print(f"[ERROR] sd.query_devices() failed: {e}")
        return

    for idx, dev in enumerate(devices):
        name = dev.get("name", "<?>")
        max_in = dev.get("max_input_channels", 0)
        max_out = dev.get("max_output_channels", 0)

        flags = []
        if max_in > 0:
            flags.append("INPUT_OK")
        if max_out > 0:
            flags.append("OUTPUT_OK")
        flag_str = ", ".join(flags) if flags else "NO_IO"

        print(f"{idx:3d}: {name:40s} | in={max_in} out={max_out} | {flag_str}")

    # Show current defaults
    try:
        default_dev = sd.default.device
    except Exception as e:
        print(f"\n[WARN] Could not read sd.default.device: {e}")
    else:
        print("\n=== defaults ===")
        print(f"sd.default.device = {default_dev}  (format: [input, output])")


if __name__ == "__main__":
    main()
