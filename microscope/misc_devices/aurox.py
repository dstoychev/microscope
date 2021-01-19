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

"""Adds support for Aurox devices.

Requires package aurox_clarity.

Device server configuration needs to be defined as a function. For example:

    #!/usr/bin/env python

    import microscope
    import microscope.testsuite.devices as testdevices
    import microscope.misc_devices.aurox as aurox
    from microscope.device_server import device

    def create_Clarity(**kwargs):
        del kwargs
        cam = testdevices.TestCamera()
        clarity = aurox.Clarity(cam)
        return {"Camera": cam, "AuroxClarity": clarity}

    DEVICES = [
        device(create_Clarity, "localhost", 8000),
    ]

"""

import logging
import enum
import typing
import time
import aurox_clarity.controller
import aurox_clarity.processor
import numpy
import microscope
import microscope.abc

_logger = logging.getLogger(__name__)


def _process_data_patched(self, data: numpy.ndarray) -> numpy.ndarray:
    if self._clarity.calibrated and self._clarity.confocal_mode:
        position = self._clarity.get_filter_position()
        if position:
            return super(self.__class__, self)._process_data(
                self._clarity._processors[position].process(data).get()
            )
    return super(self.__class__, self)._process_data(data)


def _get_sensor_shape_patched(self) -> typing.Tuple[int, int]:
    shape = self.__class__._get_sensor_shape()
    if self._clarity.calibrated and self._clarity.confocal_mode:
        return (shape[0] // 2, shape[1])
    return shape


CLARITY_CAMERA_ASSOCIATION_LABEL = "Aurox Clarity ID"


class ClarityDiskSectioning(enum.IntEnum):
    """Enumeration of disk sectioning options.

    Used for getting or setting the disk sectioning of Clarity devices.
    NOTE: low/mid/high sectioning correspond to high/mid/low signal level.
    """

    NONE = aurox_clarity.controller.DSKPOS0
    LOW = aurox_clarity.controller.DSKPOS1
    MID = aurox_clarity.controller.DSKPOS2
    HIGH = aurox_clarity.controller.DSKPOS3


class Clarity(microscope.abc.Device):
    """Aurox Clarity device.

    During initialisation the camera device will have a new setting appended.
    This setting contains the index of the Clarity device among all discovered
    such devices and it is used for associating the two. It is expected that
    this is an injective relation, i.e. one camera per Clarity device. The
    camera device is further monkey-patched to modify the way it processes
    data and reports its sensor shape.

    Args:
        camera: camera device which will be associated with the Clarity device.
        index: the index of the device in case multiple are connected.

    """

    _filter_positions = (
        aurox_clarity.controller.FLTPOS1,
        aurox_clarity.controller.FLTPOS2,
        aurox_clarity.controller.FLTPOS3,
        aurox_clarity.controller.FLTPOS4,
    )

    def __init__(
        self, camera: microscope.abc.Camera, index: int = 0, **kwargs
    ):
        super().__init__(index, **kwargs)
        self._ctrl = aurox_clarity.controller.Controller(index)
        self._processors = [None, None, None, None]
        self._confocal_mode = False
        # Assign Clarity's index as the camera's ID
        self._camera = camera
        self._camera.add_setting(
            CLARITY_CAMERA_ASSOCIATION_LABEL,
            "int",
            lambda: index,
            lambda value: None,
            (index,),
            lambda: True,
        )
        # Monkey-patch camera
        self._camera._clarity = self
        self._camera._process_data = _process_data_patched.__get__(
            self._camera, self._camera.__class__
        )
        self._camera._get_sensor_shape = _get_sensor_shape_patched.__get__(
            self._camera, self._camera.__class__
        )

    def _do_disable(self) -> None:
        self._ctrl.switchOff()

    def _do_enable(self) -> bool:
        self._ctrl.switchOn()
        return self._ctrl.getOnOff() == aurox_clarity.controller.RUN

    def _do_shutdown(self) -> None:
        pass

    def get_disk_position(self) -> typing.Optional[ClarityDiskSectioning]:
        """Get the position of the disk.

        Use the `ClarityDiskSectioning` enumeration to make sense of the
        return value. In case the disk is still sliding there is no return
        value.
        """
        dpos = self._ctrl.getDiskPosition()
        if dpos == aurox_clarity.controller.DSKERR:
            raise microscope.DeviceError("Error querying disk position.")
        if dpos == aurox_clarity.controller.DSKMID:
            return None
        return ClarityDiskSectioning(dpos)

    def set_disk_position(
        self,
        position: ClarityDiskSectioning,
        blocking: bool = False,
        timeout_s: float = 5.0,
    ) -> None:
        """Set the position of the disk.

        When blocking, the sleep interval is 1 second.

        Args:
            position: new disk position.
            blocking: switches blocking mode, in which the function waits
                for the disk to reach its destination.
            timeout_s: timeout threshold for the blocking mode. In seconds.
        """
        _ = self._ctrl.setDiskPosition(position.value)
        if blocking:
            time_start = time.time()
            # There is an initial delay between the issuing of the command and
            # the start of motion; experimentally determined to be around 50ms
            time.sleep(0.05)
            # Wait until the final destination has been reached; NOTE: during
            # motion, around 50ms are spent on any intermediate positions, so
            # the getDiskPosition() may temporarily return invalid values
            while self.get_disk_position() != position:
                # Changing the disk position could take anywhere between 1 to 3
                # seconds, depending of the distance. Experimentally determined.
                time.sleep(1)
                if time.time() - time_start >= timeout_s:
                    raise microscope.DeviceError(
                        "Timeout during changing of disk position."
                    )

    def get_filter_position(self) -> typing.Optional[int]:
        """Get the position of the filter cube turret.

        The return value is in the range [0;3]. Nothing is returned
        if the turret is still moving.
        """
        fpos = self._ctrl.getFilterPosition()
        if fpos == aurox_clarity.controller.FLTERR:
            raise microscope.DeviceError("Error querying filter position")
        if fpos == aurox_clarity.controller.FLTMID:
            return None
        return self._filter_positions.index(fpos)

    def set_filter_position(
        self, position: int, blocking: bool = False, timeout_s: float = 0.5
    ) -> None:
        """Sets the position of the filter cube turret.

        When blocking, the interval is 150 miliseconds.

        Args:
            position: new turret position.
            blocking: switches blocking mode, in which the function waits
                for the turret to reach its destination.
            timeout_s: timeout threshold for the blocking mode. In seconds.
        """
        _ = self._ctrl.setFilterPosition(self._filter_positions[position])
        if blocking:
            time_start = time.time()
            while self.get_filter_position() != position:
                # Changing the filter position could take anywhere between 120
                # and 200 miliseconds, depending on whether the turret needs to
                # move one or two positions, respectively. These values were
                # determined experimentally.
                time.sleep(0.15)
                if time.time() - time_start > timeout_s:
                    raise microscope.DeviceError(
                        "Timeout during changing of filter position."
                    )

    @property
    def calibration_led(self) -> bool:
        """Status of the calibration LED."""
        return self._ctrl.getCalibrationLED() == aurox_clarity.controller.CALON

    @calibration_led.setter
    def calibration_led(self, state: bool) -> None:
        """State of the calibration LED."""
        param = aurox_clarity.controller.CALOFF
        if state:
            param = aurox_clarity.controller.CALON
        self._ctrl.setCalibrationLED(param)

    @property
    def calibrated(self) -> bool:
        """Calibration state for the current filter cube turret position."""
        position = self.get_filter_position()
        if position:
            return self._processors[position] is not None
        return False

    def calibrate(self, data: numpy.ndarray) -> None:
        """Perform calibration routine.

        Waits for the filter cube turret to stop moving if it is.
        The calibration is valid only for that particular turret position.
        Re-calibration is needed in case the filter cube is replaced.

        Args:
            data: image of the pattern from the calibration LED.
        """
        if not self.enabled:
            _logger.warning(
                "Cannot perform calibration when Clarity is not enabled."
            )
            return
        # Ensure the turret has stopped rotating
        position = self.get_filter_position()
        while position is None:
            position = self.get_filter_position()
        try:
            self._processors[position] = aurox_clarity.processor.Processor(
                data
            )
        except Exception as e:
            raise microscope.DeviceError(
                "Failed to calibrate Clarity Device! " + repr(e)
            )

    @property
    def door_closed(self) -> bool:
        """State of the filter cube turret door."""
        return self._ctrl.getDoor() == aurox_clarity.controller.DOORCLSD

    def get_serial(self) -> int:
        """Get the serial number of the device."""
        return self._ctrl.getSerialNumber()

    def get_index(self) -> str:
        """Get the device index."""
        return self._index

    @property
    def confocal_mode(self) -> bool:
        """State of the confocal mode."""
        return self._confocal_mode

    @confocal_mode.setter
    def confocal_mode(self, state: bool) -> None:
        """State of the confocal mode."""
        self._confocal_mode = state
