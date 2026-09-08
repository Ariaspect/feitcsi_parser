#!/usr/bin/env python3
import subprocess
import time
import sys
import numpy as np
import os
import signal

# Global variable to track current status
current_status = False

##########
# threshold c a duration
##########
# out
# 40 1

# Global settings for testing factor
# threshold = 5.0  # Change detection threshold (dB)
# threshold = 10.0  # Change detection threshold (dB)
# threshold = 15.0  # Change detection threshold (dB)
threshold = 17.0  # Change detection threshold (dB)
# threshold = 18.0  # Change detection threshold (dB)
# threshold = 19.0  # Change detection threshold (dB)
# threshold = 20.0  # Change detection threshold (dB)
# threshold = 21.0  # Change detection threshold (dB)
# threshold = 23.0  # Change detection threshold (dB)
threshold = 26.0  # Change detection threshold (dB)
# threshold = 27.0  # Change detection threshold (dB)
# threshold = 28.0  # Change detection threshold (dB)
# threshold = 30.0  # Change detection threshold (dB)
# threshold = 40.0  # Change detection threshold (dB)
# threshold = 45.0  # Change detection threshold (dB)
# threshold = 50.0  # Change detection threshold (dB)
# threshold = 52.0  # Change detection threshold (dB)
# threshold = 55.0  # Change detection threshold (dB)
# threshold = 60.0  # Change detection threshold (dB)
# threshold = 100.0  # Change detection threshold (dB)

absence_duration = 10.0  # Duration to wait before reporting absence (in seconds)

min_change_subcarriers = (
    2  # Minimum number of subcarriers that need to change to detect movement
)
min_absence_subcarriers = (
    2  # Minimum number of subcarriers that need to change to absence state
)


def run_command(command):
    """General run command"""
    try:
        result = subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True
        )
        return result
    except subprocess.CalledProcessError as e:
        sys.exit(1)


def start_ping():
    """Start ping"""
    try:
        dev_null = open(os.devnull, "w")
        ping_process = subprocess.Popen(
            ["ping", "192.168.0.1"],
            stdout=dev_null,
            stderr=dev_null,
            preexec_fn=os.setsid,  # Set process group
        )
        return ping_process, dev_null
    except Exception as e:
        sys.exit(1)


def stop_ping(ping_process, dev_null):
    """Stop ping"""
    try:
        # Terminate process group
        os.killpg(ping_process.pid, signal.SIGTERM)
        ping_process.wait(timeout=5)
    except Exception:
        pass
    finally:
        dev_null.close()


def mtk_read_bf_csi(in_bytes):
    """
    Parse CSI Frame
    :param in_bytes: CSI frame data as uint8_array
    :return: CSI Dict or None if parsing fails
    """
    global csi
    timestamp_low = 0
    rssi = 0
    subcarrier = 256

    index = 0
    while index < len(in_bytes) - 1:
        field_tag = in_bytes[index]
        field_length = np.uint16(in_bytes[index + 1] + (in_bytes[index + 2] << 8))
        index += 3

        if field_tag == 0:
            version = np.uint32(in_bytes[index])
        elif field_tag == 2:
            timestamp_low = (
                np.uint64(in_bytes[index])
                + np.uint64(in_bytes[index + 1] << 8)
                + np.uint64(in_bytes[index + 2] << 16)
                + np.uint64(in_bytes[index + 3] << 24)
            )
        elif field_tag == 3:
            rssi = np.uint32(in_bytes[index])
        elif field_tag == 4:
            snr = np.uint32(in_bytes[index])
        elif field_tag == 5:
            bandwidth = np.uint32(in_bytes[index])
            if bandwidth == 0:
                subcarrier = 64
            elif bandwidth == 1:
                subcarrier = 128
            elif bandwidth == 2:
                subcarrier = 256
            csi = np.zeros(subcarrier, dtype=np.complex128)
        elif field_tag == 6:
            pci = np.uint32(in_bytes[index])
        elif field_tag == 7:
            transmitter_mac = (
                ""
                + in_bytes[index].tobytes().hex()
                + ":"
                + in_bytes[index + 1].tobytes().hex()
                + ":"
                + in_bytes[index + 2].tobytes().hex()
                + ":"
                + in_bytes[index + 3].tobytes().hex()
                + ":"
                + in_bytes[index + 4].tobytes().hex()
                + ":"
                + in_bytes[index + 5].tobytes().hex()
            )
        elif field_tag == 8:
            for i in range(subcarrier):
                value = (int(in_bytes[index + (i * 2) + 1]) << 8) | int(
                    in_bytes[index + (i * 2)]
                )
                value_14bit = value & 0x3FFF
                if value_14bit & 0x2000:
                    value_14bit = value_14bit - 0x4000
                csi.real[i] = value_14bit
        elif field_tag == 9:
            for i in range(subcarrier):
                value = (int(in_bytes[index + (i * 2) + 1]) << 8) | int(
                    in_bytes[index + (i * 2)]
                )
                value_14bit = value & 0x3FFF
                if value_14bit & 0x2000:
                    value_14bit = value_14bit - 0x4000
                csi.imag[i] = value_14bit
        elif field_tag == 10:
            extra = (
                np.uint64(in_bytes[index])
                + np.int64(in_bytes[index + 1] << 8)
                + np.uint64(in_bytes[index + 2] << 16)
                + np.uint64(in_bytes[index + 3] << 24)
            )
        elif field_tag == 15:
            tpi = np.uint32(in_bytes[index])
        elif field_tag == 16:
            rpi = np.uint32(in_bytes[index])
        elif field_tag == 17:
            frame_mode = np.uint32(in_bytes[index])
        elif field_tag == 18:
            deprecated = (
                np.uint64(in_bytes[index])
                + np.uint64(in_bytes[index + 1] << 8)
                + np.uint64(in_bytes[index + 2] << 16)
                + np.uint64(in_bytes[index + 3] << 24)
            )
        elif field_tag == 19:
            rx_rate = np.uint32(in_bytes[index])
        elif field_tag == 22:
            band = np.uint32(in_bytes[index])
        elif field_tag == 23:
            tone_valid = (
                np.uint64(in_bytes[index])
                + np.uint64(in_bytes[index + 1] << 8)
                + np.uint64(in_bytes[index + 2] << 16)
                + np.uint64(in_bytes[index + 3] << 24)
            )

        index += field_length

    csi_dict = {
        "timestamp_low": timestamp_low,
        "subcarrier": subcarrier,
        "rssi": rssi,
        "csi": csi,
    }
    return csi_dict


def process_csi_data(frame_data, previous_magnitude_db):
    """
    Process a single CSI frame and detect changes
    :param frame_data: Single CSI frame data (without Magic Number and length)
    :param previous_magnitude_db: Previous frame's magnitude (dB)
    :return: Tuple of (Updated previous_magnitude_db, True if movement detected, False otherwise)
    """
    global current_status  # Use global variable

    try:
        csi_dict = mtk_read_bf_csi(frame_data)
    except Exception:
        return previous_magnitude_db, False

    magnitude = np.abs(csi_dict["csi"])
    magnitude_db = 20 * np.log10(magnitude + 1e-10)

    # Skip change detection if subcarrier count differs and update previous_magnitude_db
    if (
        previous_magnitude_db is not None
        and magnitude_db.shape != previous_magnitude_db.shape
    ):
        return magnitude_db.copy(), False

    # Detect changes
    movement_detected = False
    if previous_magnitude_db is not None:
        diff = np.abs(magnitude_db - previous_magnitude_db)
        if not current_status:  # absence state
            change_indices = np.where(diff > threshold)[0]
        else:  # presence state
            change_indices = np.where(diff > (threshold - 1))[0]

        # Check if the number of changed subcarriers exceeds the minimum threshold
        min_count = 0
        if not current_status:  # absence state
            min_count = min_change_subcarriers
        else:  # presence state
            min_count = min_absence_subcarriers

        if len(change_indices) >= min_count:
            # Notice time to change into present state
            if not current_status:
                time.sleep(2)
            print("+", flush=True)
            current_status = True  # Update current status to True
            movement_detected = True

    # Update previous frame
    previous_magnitude_db = magnitude_db.copy()

    return previous_magnitude_db, movement_detected


def main(capture_time_sec):
    """Capture and process CSI raw data in real-time"""
    global current_status  # Use global variable
    current_status = False  # Initialize current status to False

    # Configure WiFi chip
    run_command("/home/iwpriv wlan0 driver 'set_csi 2 0 1'")
    run_command("/home/iwpriv wlan0 driver 'set_csi 2 3 0 34'")
    run_command("/home/iwpriv wlan0 driver 'set_csi 2 5 2'")

    # Start ping
    ping_process, dev_null = start_ping()

    # Start CSI capture
    run_command("/home/iwpriv wlan0 driver 'set_csi 1'")

    # Start cat process
    cat_process = subprocess.Popen(
        ["/bin/cat", "/proc/net/wlan/csi_data"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,  # Set process group
    )

    # Real-time data processing
    previous_magnitude_db = None
    start_time = time.time()
    data_buffer = b""
    last_movement_time = None  # Last time movement was detected
    absence_reported = False  # Flag to check if absence state (-) has been reported

    try:
        while time.time() - start_time < capture_time_sec:
            # Read data from cat's stdout
            chunk = cat_process.stdout.read(1024)  # Read 1024 bytes at a time
            if chunk:
                data_buffer += chunk

            # Split frames by 0xAC
            data = np.frombuffer(data_buffer, dtype=np.uint8)
            frame_start = 0
            while frame_start < len(data):
                # Find Magic Number (0xAC)
                magic_pos = data[frame_start:].tobytes().find(b"\xac")
                if magic_pos == -1:  # No Magic Number found
                    data_buffer = data[frame_start:].tobytes()  # Save remaining data
                    break

                frame_start += magic_pos
                if frame_start + 2 >= len(data):  # Cannot read length field
                    data_buffer = data[frame_start:].tobytes()
                    break

                # Read frame length
                length = int.from_bytes(
                    data[frame_start + 1 : frame_start + 3], byteorder="little"
                )
                if frame_start + length > len(data):  # Frame is incomplete
                    data_buffer = data[frame_start:].tobytes()
                    break

                # Extract frame data (excluding Magic Number and length)
                frame_data = data[frame_start + 3 : frame_start + length]
                frame_start += length

                # Process frame
                previous_magnitude_db, movement_detected = process_csi_data(
                    frame_data, previous_magnitude_db
                )

                # Update time on movement detection
                if movement_detected:
                    last_movement_time = time.time()
                    absence_reported = (
                        False  # Switch to presence state, reset absence report flag
                    )

            # Switch to absence state if no movement for the specified duration
            current_time = time.time()
            if last_movement_time is not None and not absence_reported:
                if current_time - last_movement_time >= absence_duration:
                    print("-", flush=True)
                    current_status = False  # Update current status to False
                    absence_reported = True  # Absence state reported

    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        pass  # Ignore broken pipe exception
    finally:
        # Stop CSI capture
        try:
            run_command("/home/iwpriv wlan0 driver 'set_csi 0'")
        except:
            pass

        # Stop ping
        stop_ping(ping_process, dev_null)

        # Terminate cat process
        try:
            os.killpg(cat_process.pid, signal.SIGTERM)
            cat_process.wait(timeout=5)
        except:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    try:
        capture_time = int(sys.argv[1])
        if capture_time <= 0:
            raise ValueError
    except ValueError:
        sys.exit(1)

    main(capture_time)

