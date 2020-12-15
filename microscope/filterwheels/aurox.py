#!/usr/bin/env python3

## Copyright (C) 2020 Mick Phillips <mick.phillips@gmail.com>
##
## This file is part of Microscope.
##
## Microscope is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Microscope is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Microscope.  If not, see <http://www.gnu.org/licenses/>.

"""Adds support for Aurox devices

Requires package hidapi.

Config sample:

device(microscope.filterwheels.aurox.Clarity,
       {'camera': 'microscope.Cameras.cameramodule.SomeCamera',
        'camera.someSetting': value})

Deconvolving data requires:
 * availability of clarity_process and cv2
 * successful completion of a calibration step
   + set_mode(Modes.calibrate)
   + trigger the camera to generate an image
   + when the camera returns the image, calibration is complete
"""

import time
from threading import Lock
import typing
import enum
import logging
import hid
import microscope
import microscope.devices

_logger = logging.getLogger(__name__)

try:
    # Currently, clarity_process is a module that is not packaged, so needs
    # to be put on the python path somewhere manually.
    from clarity_process import ClarityProcessor
except Exception:
    _logger.warning(
        "Could not import clarity_process module:" "no processing available."
    )

Mode = enum.IntEnum("Mode", "difference, raw, calibrate")

# Clarity constants. These may differ across products, so mangle names.
# USB IDs
_Clarity__VENDORID = 0x1F0A
_Clarity__PRODUCTID = 0x0088
# Base status
_Clarity__SLEEP = 0x7F
_Clarity__RUN = 0x0F
# Door status
_Clarity__DOOROPEN = 0x01
_Clarity__DOORCLOSED = 0x02
# Disk position/status
_Clarity__SLDPOS0 = 0x00  # disk out of beam path, wide field
_Clarity__SLDPOS1 = 0x01  # disk pos 1, low sectioning
_Clarity__SLDPOS2 = 0x02  # disk pos 2, mid sectioning
_Clarity__SLDPOS3 = 0x03  # disk pos 3, high sectioning
_Clarity__SLDERR = 0xFF  # An error has occurred in setting slide position (end stops not detected)
_Clarity__SLDMID = 0x10  # slide between positions (was =0x03 for SD62)
# Filter position/status
_Clarity__FLTPOS1 = 0x01  # Filter in position 1
_Clarity__FLTPOS2 = 0x02  # Filter in position 2
_Clarity__FLTPOS3 = 0x03  # Filter in position 3
_Clarity__FLTPOS4 = 0x04  # Filter in position 4
_Clarity__FLTERR = 0xFF  # An error has been detected in the filter drive (eg filters not present)
_Clarity__FLTMID = 0x10  # Filter between positions
# Calibration LED state
_Clarity__CALON = 0x01  # CALibration led power on
_Clarity__CALOFF = 0x02  # CALibration led power off
# Error status
_Clarity__CMDERROR = 0xFF  # Reply to a command that was not understood
# Commands
_Clarity__GETVERSION = 0x00  # Return 3-byte version number byte1.byte2.byte3
# State commands: single command byte immediately followed by any data.
_Clarity__GETONOFF = 0x12  # No data out, returns 1 byte on/off status
_Clarity__GETDOOR = 0x13  # No data out, returns 1 byte shutter status, or SLEEP if device sleeping
_Clarity__GETSLIDE = 0x14  # No data out, returns 1 byte disk-slide status, or SLEEP if device sleeping
_Clarity__GETFILT = 0x15  # No data out, returns 1 byte filter position, or SLEEP if device sleeping
_Clarity__GETCAL = 0x16  # No data out, returns 1 byte CAL led status, or SLEEP if device sleeping
_Clarity__GETSERIAL = (
    0x19  # No data out, returns 4 byte BCD serial number (little endian)
)
_Clarity__FULLSTAT = 0x1F  # No data, Returns 10 bytes VERSION[3],ONOFF,SHUTTER,SLIDE,FILT,CAL,??,??
# Run state action commands
_Clarity__SETONOFF = 0x21  # 1 byte out on/off status, echoes command or SLEEP
_Clarity__SETSLIDE = 0x23  # 1 byte out disk position, echoes command or SLEEP
_Clarity__SETFILT = 0x24  # 1 byte out filter position, echoes command or SLEEP
_Clarity__SETCAL = 0x25  # 1 byte out CAL led status, echoes command or SLEEP
# Service mode commands. Stops disk spinning for alignment.
_Clarity__SETSVCMODE1 = 0xE0  # 1 byte for service mode. SLEEP activates service mode. RUN returns to normal mode.


class _CameraAugmentor:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aurox_mode = Mode.raw
        self._processor = None

    def set_aurox_mode(self, mode):
        self._aurox_mode = mode

    def _process_data(self, data):
        """Process data depending on state of self._aurox_mode."""
        if self._aurox_mode == Mode.raw:
            return data
        elif self._aurox_mode == Mode.difference:
            if self._processor is None:
                raise Exception("Not calibrated yet - can not process image")
            return self._processor.process(data).get()
        elif self._aurox_mode == Mode.calibrate:
            # This will introduce a significant delay, but returning the
            # image indicates that the calibration step is complete.
            self._processor = ClarityProcessor(data)
            return data
        else:
            raise Exception("Unrecognised mode: %s", self._aurox_mode)

    def get_sensor_shape(self):
        """Return image shape accounting for rotation and Aurox processing."""
        shape = self._get_sensor_shape()
        # Does current mode combine two halves into a single image?
        if self._aurox_mode in [Mode.difference]:
            shape = (shape[1] // 2, shape[0])
        # Does the current transform perform a 90-degree rotation?
        if self._transform[2]:
            # 90 degree rotation
            shape = (shape[1], shape[0])
        return shape


class Clarity(
    microscope.devices.ControllerDevice, microscope.devices.FilterWheelBase
):
    """Adds support for Aurox Clarity

    Acts as a ControllerDevice providing the camera attached to the Clarity."""

    _slide_to_sectioning = {
        __SLDPOS0: "bypass",
        __SLDPOS1: "low",
        __SLDPOS2: "mid",
        __SLDPOS3: "high",
    }
    _positions = 4
    _resultlen = {
        __GETONOFF: 1,
        __GETDOOR: 1,
        __GETSLIDE: 1,
        __GETFILT: 1,
        __GETCAL: 1,
        __GETSERIAL: 4,
        __FULLSTAT: 10,
    }
    _filters = (
        __FLTPOS1,
        __FLTPOS2,
        __FLTPOS3,
        __FLTPOS4,
    )

    def __init__(self, camera=None, camera_kwargs={}, **kwargs) -> None:
        """Create a Clarity instance controlling an optional Camera device.

        :param camera: a class to control the connected camera
        :param camera_kwargs: parameters passed to camera as keyword arguments
        """
        super().__init__(positions=Clarity._positions, **kwargs)
        self._lock = Lock()
        self._hid = None
        self._devices = {}
        if camera is None:
            self._cam = None
            _logger.warning("No camera specified.")
            self._can_process = False
        else:
            AugmentedCamera = type(
                "AuroxAugmented" + camera.__name__,
                (_CameraAugmentor, camera),
                {},
            )
            self._cam = AugmentedCamera(**camera_kwargs)
            self._can_process = "ClarityProcessor" in globals()
        # Acquisition mode
        self._mode = Mode.raw
        # Cached filter and slide positions
        self._cached_slide = __SLDPOS0
        self._cached_filter = __FLTPOS1
        # Add device settings
        self.add_setting(
            "sectioning",
            "enum",
            self.get_slide_position,
            lambda val: self.set_slide_position(val),
            self._slide_to_sectioning,
        )
        self.add_setting(
            "mode", "enum", lambda: self._mode.name, self.set_mode, Mode
        )

    @property
    def devices(self) -> typing.Mapping[str, microscope.devices.Device]:
        """Devices property, required by ControllerDevice interface."""
        if self._cam:
            return {"camera": self._cam}
        else:
            return {}

    def set_mode(self, mode: Mode) -> None:
        """Set the operation mode"""
        if mode in [Mode.calibrate, Mode.difference] and not self._can_process:
            raise Exception("Processing not available")
        else:
            self._cam.set_aurox_mode(mode)
        if mode == Mode.calibrate:
            self._set_calibration(True)
        else:
            self._set_calibration(False)

    def _send_command(self, command, param=0, max_length=16, timeout_ms=100):
        """Send a command to the Clarity and return its response"""
        if not self._hid:
            self.open()
        with self._lock:
            # The device expects a list of 16 integers
            buffer = [0x00] * max_length  # The 0th element must be 0.
            buffer[1] = command  # The 1st element is the command
            buffer[2] = param  # The 2nd element is any command argument.
            result = self._hid.write(buffer)
            if result == -1:
                # Nothing to read back. Check hid error state.
                err = self._hid.error()
                if err != "":
                    self.close()
                    raise microscope.DeviceError(err)
                else:
                    return None
            while True:
                # Read responses until we see the response to our command.
                # (We should get the correct response on the first read.)
                response = self._hid.read(result - 1, timeout_ms)
                if not response:
                    # No response
                    return None
                elif response[0] == command:
                    break
            bytes = self._resultlen.get(command, None)
            if bytes is None:
                return response[1:]
            elif bytes == 1:
                return response[1]
            else:
                return response[1:]

    @property
    def is_connected(self):
        return self._hid is not None

    def open(self):
        h = hid.device()
        h.open(vendor_id=__VENDORID, product_id=__PRODUCTID)
        h.set_nonblocking(False)
        self._hid = h

    def close(self):
        if self.is_connected:
            self._hid.close()
            self._hid = None

    def get_id(self):
        return self._send_command(__GETSERIAL)

    def _do_enable(self):
        if not self.is_connected:
            self.open()
        self._send_command(__SETONOFF, __RUN)
        return self._send_command(__GETONOFF) == __RUN

    def _do_disable(self):
        self._send_command(__SETONOFF, __SLEEP)

    def _set_calibration(self, state):
        if state:
            result = self._send_command(__SETCAL, __CALON)
        else:
            result = self._send_command(__SETCAL, __CALOFF)
        return result

    def get_slide_position(self):
        """Get the current slide position"""
        result = self._send_command(__GETSLIDE)
        if result is None or result == __SLDERR:
            raise microscope.DeviceError("Slide position error.")
        elif result != __SLDMID:
            self._cached_slide = result
        return self._cached_slide

    def set_slide_position(self, position, blocking=True):
        """Set the slide position"""
        result = self._send_command(__SETSLIDE, position)
        if result is None:
            raise microscope.DeviceError("Slide position error.")
        self._cached_slide = pos
        if blocking:
            # Initial delay
            time.sleep(0.05)
            while True:
                # Wait for 3 consecutive non-mid positions, to avoid spurious
                # false negatives
                moving = []
                for _ in range(3):
                    moving.append(self._send_command(__GETSLIDE) == __SLDMID)
                    time.sleep(0.01)
                if any(moving):
                    break

    def get_slides(self):
        return self._slide_to_sectioning

    def get_status(self):
        # A status dict to populate and return
        status = dict.fromkeys(
            [
                "connected",
                "on",
                "door open",
                "slide",
                "filter",
                "calibration",
                "busy",
                "mode",
            ]
        )
        status["mode"] = self._mode.name
        # Fetch 10 bytes VERSION[3],ONOFF,SHUTTER,SLIDE,FILT,CAL,??,??
        try:
            result = self._send_command(__FULLSTAT)
            status["connected"] = True
        except Exception:
            status["connected"] = False
            return status
        # A list to track states, any one of which mean the device is busy.
        busy = []
        # Disk running
        status["on"] = result[3] == __RUN
        # Door open
        # Note - it appears that the __DOOROPEN and __DOORCLOSED status states
        # are switched, or that the DOOR is in fact an internal shutter. I'll
        # interpret 'door' as the external door here, as that is what the user
        # can see. When the external door is open, result[4] == __DOORCLOSED
        door = result[4] == __DOORCLOSED
        status["door open"] = door
        busy.append(door)
        # Slide position
        slide = result[5]
        if slide == __SLDMID:
            # Slide is moving
            status["slide"] = (None, "moving")
            busy.append(True)
        else:
            status["slide"] = (
                slide,
                self._slide_to_sectioning.get(slide, None),
            )
        # Filter position
        filter = result[6]
        if filter == __FLTMID:
            # Filter is moving
            status["filter"] = (None, "moving")
            busy.append(True)
        else:
            status["filter"] = result[6]
        # Calibration LED on
        status["calibration"] = result[7] == __CALON
        # Slide or filter moving
        status["busy"] = any(busy)
        return status

    # Implemented by FilterWheelBase
    # def get_filters(self):
    #    pass

    def _do_get_position(self):
        """Return the current filter position"""
        result = self._send_command(__GETFILT)
        if result is None or result == __FLTERR:
            raise microscope.DeviceError("Filter position error.")
        elif result != __FLTMID:
            self._cached_filter = self._filters.index(result)
        return self._cached_filter

    def _do_set_position(self, pos, blocking=True):
        """Set the filter position"""
        result = self._send_command(__SETFILT, self._filters[pos])
        if result is None:
            raise microscope.DeviceError("Filter position error.")
        self._cached_filter = pos
        if blocking:
            # Initial delay
            time.sleep(0.05)
            while True:
                # Wait for 3 consecutive non-mid positions, to avoid spurious
                # false negatives
                moving = []
                for _ in range(3):
                    moving.append(self._send_command(__GETFILT) == __FLTMID)
                    time.sleep(0.01)
                if any(moving):
                    break
