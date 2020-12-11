#!/usr/bin/env python3

## Copyright (C) 2020 David Miguel Susano Pinto <carandraug@gmail.com>
## Copyright (C) 2020 Ian Dobbie <ian.dobbie@bioch.ox.ac.uk>
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

"""Olympus IX-TPC.

The IX3 Touch Panel Controller (TPC) comes with the IX83 microscope
frames and it controls all motorised components via the IX3-CBH 
control unit. This software implementation is a wrapper around the
PortManager software library, which is proprietary and can be
requested from Olympus.

"""
import logging
import ctypes
import sys
import pathlib
import typing
import enum
import dataclasses
import microscope
import microscope._utils
import microscope.abc
import time


_logger = logging.getLogger(__name__)


class _MDK_MSL_CMD(ctypes.Structure):
    _SZ_SMALL = 8
    _MAX_TAG_SIZE = 32
    _MAX_COMMAND_SIZE = 256
    _MAX_RESPONSE_SIZE = 256
    _fields_ = [
        # command block signature
        ("m_Signature", ctypes.c_ulong),
        # Basic fields
        # unit type
        ("m_Type", ctypes.c_int),
        # command sequence
        ("m_Sequence", ctypes.c_ushort),
        # from
        ("m_From", ctypes.c_char * _SZ_SMALL),
        # to
        ("m_To", ctypes.c_char * _SZ_SMALL),
        # command status
        ("m_Status", ctypes.c_int),
        # result
        ("m_Result", ctypes.c_ulong),
        # sync or async?
        ("m_Sync", ctypes.c_long),
        # command or query? (refer to command tag)
        ("m_Command", ctypes.c_long),
        # TRUE means NOT wait response
        ("m_SendOnly", ctypes.c_long),
        # Management fields
        # start time
        ("m_StartTime", ctypes.c_longlong),
        # end time
        ("m_FinishTime", ctypes.c_longlong),
        # time out (ms)
        ("m_Timeout", ctypes.c_ulong),
        # callback entry
        ("m_Callback", ctypes.c_void_p),
        # event info
        ("m_Event", ctypes.c_void_p),
        # context
        ("m_Context", ctypes.c_void_p),
        # timer ID
        ("m_TimerID", ctypes.c_uint),
        # port info
        ("m_PortContext", ctypes.c_void_p),
        # Extension fields (for individual commands)
        # extend info 1
        ("m_Ext1", ctypes.c_ulong),
        # extend info 2
        ("m_Ext2", ctypes.c_ulong),
        # extend info 3
        ("m_Ext3", ctypes.c_ulong),
        # LONGLONG extend info 1
        ("m_lExt1", ctypes.c_longlong),
        # LONGLONG extend info 2
        ("m_lExt2", ctypes.c_longlong),
        # Tag fields
        # tag size
        ("m_TagSize", ctypes.c_ulong),
        ("m_CmdTag", ctypes.c_byte * _MAX_TAG_SIZE),
        # Command string
        # size of command string
        ("m_CmdSize", ctypes.c_ulong),
        ("m_Cmd", ctypes.c_byte * _MAX_COMMAND_SIZE),
        # Response string
        # size of response string
        ("m_RspSize", ctypes.c_ulong),
        ("m_Rsp", ctypes.c_byte * _MAX_RESPONSE_SIZE),
    ]


_CALLBACK_TYPE = ctypes.CFUNCTYPE(
    # return type
    ctypes.c_int,
    # MsgId
    ctypes.c_ulong,
    # wParam
    ctypes.c_ulong,
    # lParam
    ctypes.c_ulong,
    # pv
    ctypes.c_void_p,
    # pContext
    ctypes.c_void_p,
    # pCaller
    ctypes.c_void_p,
)

# msl_pd_1394.dll needs to be manually loaded before msl_pm.dll, but other
# than being an unconventional dependency it is not used for anything; the
# other DLLs are automatically loaded as regular dependencies
ctypes.CDLL(
    str(pathlib.Path(sys.exec_prefix).joinpath("gtlib", "msl_pd_1394.dll"))
)

_dll = ctypes.CDLL(
    str(pathlib.Path(sys.exec_prefix).joinpath("gtlib", "msl_pm.dll"))
)

# Configure the library
_dll.MSL_PM_Initialize.argtypes = None
_dll.MSL_PM_Initialize.restype = ctypes.c_int
_dll.MSL_PM_EnumInterface.argtypes = None
_dll.MSL_PM_EnumInterface.restype = ctypes.c_int
_dll.MSL_PM_GetInterfaceInfo.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_void_p),
]
_dll.MSL_PM_GetInterfaceInfo.restype = ctypes.c_int
_dll.MSL_PM_OpenInterface.argtypes = [ctypes.c_void_p]
_dll.MSL_PM_OpenInterface.restype = ctypes.c_bool
_dll.MSL_PM_CloseInterface.argtypes = [ctypes.c_void_p]
_dll.MSL_PM_CloseInterface.restype = ctypes.c_bool
_dll.MSL_PM_RegisterCallback.argtypes = [
    ctypes.c_void_p,
    _CALLBACK_TYPE,
    _CALLBACK_TYPE,
    _CALLBACK_TYPE,
    ctypes.c_void_p,
]
_dll.MSL_PM_RegisterCallback.restype = ctypes.c_bool
_dll.MSL_PM_SendCommand.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(_MDK_MSL_CMD),
]
_dll.MSL_PM_SendCommand.restype = ctypes.c_bool


class CommandStatus(enum.IntEnum):
    """Enumeration of command statuses.

    The default status is WAIT and it indicates the the command has not
    completed yet. Once the callback is called, the status is resolved to
    either TIMEOUT or SUCCESS, depending on the state of the pContext
    argument.
    """

    WAIT = 0
    SUCCESS = 1
    TIMEOUT = 2


@dataclasses.dataclass
class _CommandTableEntry:
    """An encapsulation for the contents of an entry in the command table."""

    status: CommandStatus
    data: _MDK_MSL_CMD
    callback: typing.Optional[typing.Callable[..., None]] = None
    callback_args: typing.Tuple[typing.Any, ...] = ()


@_CALLBACK_TYPE
def _callback_command(
    MsgId: int,
    wParam: int,
    lParam: int,
    pv: typing.Optional[int],
    pContext: typing.Optional[int],
    pCaller: typing.Optional[int],
) -> int:
    cmd_data = ctypes.cast(pv, ctypes.POINTER(_MDK_MSL_CMD)).contents
    # ctypes NULL pointers are converted to None type, so the "or"
    # operator is used to ensure the key is an integer
    key = cmd_data.m_Callback or 0
    # Derive the response
    response = "".join([chr(x) for x in cmd_data.m_Rsp[0 : cmd_data.m_RspSize]])
    # Derive the command table
    command_table = ctypes.cast(cmd_data.m_Context, ctypes.py_object).value
    # Set the status (on timeout the pContext pointer is NULL)
    if pContext:
        command_table[key].status = CommandStatus.SUCCESS
    else:
        command_table[key].status = CommandStatus.TIMEOUT
    # Call the callback if there is one
    if command_table[key].callback:
        command_table[key].callback(
            command_table[key].status,
            response,
            *command_table[key].callback_args,
        )
    # Delete the entry from the command_table
    del command_table[key]
    return 0


@_CALLBACK_TYPE
def _callback_notify(
    MsgId: int,
    wParam: int,
    lParam: int,
    pv: typing.Optional[int],
    pContext: typing.Optional[int],
    pCaller: typing.Optional[int],
) -> int:
    # TODO: Add the ability to register/unregister handlers for given
    # type of notification. IX3TPC needs a new dictionary, similar to the
    # command table, whose keys are notification types (e.g. NFP) and
    # whose values are lists of callables. Then the register/unregister
    # methods can simply add or remove from this dictionary. In addition,
    # notifications need to be enabled during TPC configuration.
    message = ctypes.cast(pv, ctypes.c_char_p).value.decode("utf-8")
    _logger.warning("Received notification: '{:s}'.".format(message))
    return 0


@_CALLBACK_TYPE
def _callback_error(
    MsgId: int,
    wParam: int,
    lParam: int,
    pv: typing.Optional[int],
    pContext: typing.Optional[int],
    pCaller: typing.Optional[int],
) -> int:
    raise microscope.MicroscopeError(
        "PortManager error. Ensure IEEE 1394 "
        "cable has not been disconnected."
    )


class IX3TPC(microscope.abc.Controller):
    """IX3 Touch Panel Controller"""

    _COMMAND_TERMINATOR = "\r\n"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._devices: typing.Dict[str, microscope.abc.Device] = {}
        self._pm_if_data_addr = ctypes.c_void_p(0)
        self._command_table: typing.Dict[int, _CommandTableEntry] = {}
        # Initialise the PortManager library and establish connection with TPC
        self._initialise_portmanager()
        self._shutting_down = False
        # Login to the TPC and configure it
        self._login()
        self._configure()
        # Add devices
        self._devices["LHLEDC"] = _IX3LHLEDC(self)

    @property
    def devices(self) -> typing.Dict[str, microscope.abc.Device]:
        return self._devices

    def _do_shutdown(self) -> None:
        # Command callbacks stop working in __del__ and this is valid for
        # both the Controller and its devices; probably something to do with
        # threading and scheduling. The first command from all __del__'s
        # combined will have a proper response, albeit no callback, but all
        # consequent commands will have no reaction until after the python
        # program terminates, at which point the callbacks will be called
        # in chronological order of the backlog, but there will be no
        # response and no context will be set (i.e. they will appear as
        # timeouts). Therefore, the only solution seems to be to blindly
        # add delays, large enough to ensure the commands are completed,
        # and hope for the best. Even better not to send any command from
        # _on_disable and _on_shutdown in subdevices.
        self._shutting_down = True
        super()._do_shutdown()
        # Logout. Timeout value could be as low as possible, the command is
        # going to timeout anyway.
        self.send_command("L 0,0", timeout_ms=0)
        # Close the interface
        _ = _dll.MSL_PM_CloseInterface(self._pm_if_data_addr)

    def _initialise_portmanager(self) -> None:
        # Initialise
        status = _dll.MSL_PM_Initialize()
        if status != 0:
            raise microscope.InitialiseError(
                "MSL_PM_Initialize() failed with code {:d}".format(status)
            )
        # Enumerate interfaces
        count = _dll.MSL_PM_EnumInterface()
        if count == 0:
            raise microscope.InitialiseError(
                "Couldn't find active interfaces on the IEEE 1394 bus. "
                "Please ensure that the computer is connected to an "
                "IX3-CBH control unit which is powered."
            )
        if count > 1:
            raise microscope.UnsupportedFeatureError(
                "The ability to communicate with multiple devices on the "
                "IEEE 1394 bus has not been implemented yet."
            )
        # Get the address of the library's Interface object with index 0
        self._pm_if_data_addr = ctypes.c_void_p(0)
        _ = _dll.MSL_PM_GetInterfaceInfo(0, ctypes.byref(self._pm_if_data_addr))
        # Open the interface
        success = _dll.MSL_PM_OpenInterface(self._pm_if_data_addr)
        if not success:
            raise microscope.InitialiseError(
                "Failed to open interface for object at "
                "address 0x{:X}".format(self._pm_if_data_addr.value)
            )
        _logger.info(
            "Opened interface for object at address 0x{:X}".format(
                self._pm_if_data_addr.value
            )
        )
        # Register callback functions and set their context to point to self
        success = _dll.MSL_PM_RegisterCallback(
            self._pm_if_data_addr,
            _callback_command,
            _callback_notify,
            _callback_error,
            ctypes.c_void_p(id(self)),
        )
        if not success:
            raise microscope.InitialiseError(
                "Failed to register callbacks for object at "
                "address 0x{:X}".format(self._pm_if_data_addr.value)
            )

    def send_command(
        self,
        cmd: str,
        callback: typing.Optional[typing.Callable[..., None]] = None,
        callback_args: typing.Tuple[typing.Any, ...] = (),
        timeout_ms: int = 10000,
    ) -> None:
        """Send an asynchronouse command.

        Only the body of the command should be specified, without the header
        or the terminator. The underlying data associated with the command or
        the callback kwargs is deleted after the callback terminates.

        .. note::
            Callbacks can raise their own errors, but this will print an
            "Exception ignored on calling ctypes callback function: ..."
            line immediately before the traceback. This should not a point
            of concern.

        """
        # Find a new key, i.e. an unsigned integer that can fit in a pointer.
        key = 0
        keys = self._command_table.keys()
        pointer_size_bits = ctypes.sizeof(ctypes.c_void_p) * 8
        for i in range(0, 2 ** (pointer_size_bits)):
            if i not in keys:
                key = i
                break
        # Create a new entry in the command table.
        # The m_Callback field seems like the most appropriate place to
        # store the key; the m_Context field is used despite the presence
        # of a broader context (pContext argument of callbacks, set during
        # callback registration) because on timeouts this broader context
        # is not available whereas the m_Context field is always present.
        self._command_table[key] = _CommandTableEntry(
            CommandStatus.WAIT,
            _MDK_MSL_CMD(
                m_CmdSize=len(cmd) + len(self._COMMAND_TERMINATOR),
                m_Cmd=(ctypes.c_byte * _MDK_MSL_CMD.m_Cmd.size)(
                    *[ord(x) for x in cmd + self._COMMAND_TERMINATOR]
                ),
                m_Callback=key,
                m_Context=ctypes.c_void_p(id(self._command_table)),
                m_Timeout=timeout_ms,
                m_Command=True,
            ),
            callback,
            callback_args,
        )
        # Send the command via PortManager and log failures
        pmsc_success = _dll.MSL_PM_SendCommand(
            self._pm_if_data_addr,
            ctypes.byref(self._command_table[key].data),
        )
        if not pmsc_success:
            _logger.error(
                "MSL_PM_SendCommand() failed for command '{:s}'.".format(cmd)
            )

    def send_command_blocking(
        self, cmd: str, timeout_ms: int = 10000, sleep_interval: float = 0.2
    ) -> typing.Tuple[CommandStatus, str]:
        """Send a pseudo-synchronous command.

        The function uses an internal callback and waits for it to be called
        or for the command to timeout, whichever happens first. The command
        should be specified as just the body, without header or terminator.

        """
        status = CommandStatus.WAIT
        response = ""

        def _callback(status_local, response_local):
            nonlocal status, response
            status = status_local
            response = response_local

        self.send_command(cmd, callback=_callback, timeout_ms=timeout_ms)
        # Sleep for as many intervals fit into the timeout. An extra interval
        # is added to account for the truncation when converting to int.
        for _ in range(0, int(timeout_ms / 1000 / sleep_interval) + 1):
            if status != CommandStatus.WAIT:
                break
            time.sleep(sleep_interval)
        return status, response

    def _login(self) -> None:
        # TODO: Login sequence can fail if one of the units has an undefined
        # state, e.g. the nosepiece position is in-between objectives. All
        # relevant primary commands should be used to query and verify the
        # state of the microscope, i.e. "OB1?", "MU1?", "MU2?", "ESH1?", etc.
        command_sequence = (
            # Attempt to login; will succeed for window statuses 1 and 2
            ("L 1,0", "L +"),
            # Enable focus (nosepiece) and XY (stage) control
            ("EN6 1,1", "EN6 +"),
            # Enable the touch panel => Expect pACK
            ("EN5 1", "EN5 +"),
            # Switch to remote IDLE state
            ("OPE 0", "OPE +"),
        )
        for (cmd_msg, cmd_exprsp) in command_sequence:
            status, response = self.send_command_blocking(cmd_msg)
            if status != CommandStatus.SUCCESS or response != cmd_exprsp:
                raise microscope.InitialiseError(
                    "Error logging in. Command: '{:s}'. Status: '{:s}'. "
                    "Response: '{:s}'.".format(cmd_msg, status.name, response)
                )

    def _configure(self) -> None:
        # TODO: IMPORTANT !!! The DIA light enable/disable status is maintained, so the software should ensure that the shutter is closed!
        command_sequence = (
            # Turn off the screen illumination
            ("TPIL 0", "TPIL +"),
        )
        for (cmd_msg, cmd_exprsp) in command_sequence:
            status, response = self.send_command_blocking(cmd_msg)
            if status != CommandStatus.SUCCESS or response != cmd_exprsp:
                raise microscope.InitialiseError(
                    "Error configuring. Command: '{:s}'. Status: '{:s}'. "
                    "Response: '{:s}'.".format(cmd_msg, status.name, response)
                )


class _IX3LHLEDC(
    microscope._utils.OnlyTriggersBulbOnSoftwareMixin,
    microscope.abc.LightSource,
):
    def __init__(self, tpc: IX3TPC) -> None:
        super().__init__()
        self._tpc = tpc

    def _do_enable(self) -> None:
        status, response = self._tpc.send_command_blocking("DSH 0")
        if status != CommandStatus.SUCCESS or response != "DSH +":
            raise ValueError(
                "Unexpected response for command 'DSH 0'. "
                "Status: {:s}. Response: {:s}".format(status.name, response)
            )

    def _do_disable(self) -> None:
        # Controller devices are garbage collected after the Controller
        # itself, so ensure that commands are sent only if the connection
        # is still open
        if not self._tpc._shutting_down:
            status, response = self._tpc.send_command_blocking("DSH 1")
            if status != CommandStatus.SUCCESS or response != "DSH +":
                raise ValueError(
                    "Unexpected response for command 'DSH 1'. "
                    "Status: {:s}. Response: {:s}".format(status.name, response)
                )

    def _do_shutdown(self):
        pass

    def initialize(self):
        pass

    def get_status(self) -> typing.List[str]:
        return []

    def get_is_on(self):
        status, response = self._tpc.send_command_blocking("DSH?")
        if status != CommandStatus.SUCCESS or response not in (
            "DSH 0",
            "DSH 1",
        ):
            raise ValueError(
                "Unexpected response for command 'DSH?'. "
                "Status: {:s}. Response: {:s}".format(status.name, response)
            )
        return bool(int(response.split()[-1]))

    def _do_get_power(self) -> float:
        status, response = self._tpc.send_command_blocking("DSH?")
        power_int = int(response.split()[-1]) if response else -1
        if status != CommandStatus.SUCCESS or not 0 <= power_int <= 255:
            raise ValueError(
                "Unexpected response for command 'DSH?'. "
                "Status: {:s}. Response: {:s}".format(status.name, response)
            )
        return power_int / 255

    def _do_set_power(self, power: float) -> None:
        command = "DIL1 {:d}".format(int(power * 255))
        status, response = self._tpc.send_command_blocking(command)
        if status != CommandStatus.SUCCESS or response != "DIL1 +":
            raise ValueError(
                "Unexpected response for command '{:s}'. "
                "Status: {:s}. Response: {:s}".format(
                    command, status.name, response
                )
            )


class _IX3SSUAxis(microscope.abc.StageAxis):
    def __init__(self, tpc: IX3TPC) -> None:
        super().__init__()
        self._tpc = tpc

    def move_by(self, delta: float) -> None:
        pass

    def move_to(self, pos: float) -> None:
        pass

    @property
    def position(self) -> float:
        return 0

    @property
    def limits(self) -> microscope.AxisLimits:
        return microscope.AxisLimits(lower=0, upper=0)


class _IX3SSU(microscope.abc.Stage):
    def __init__(self, tpc: IX3TPC) -> None:
        super().__init__()
        self._tpc = tpc
        self._axes = {}

    def initialize(self) -> None:
        super().initialize()

    def _do_shutdown(self) -> None:
        super()._do_shutdown()

    @property
    def axes(self) -> typing.Mapping[str, microscope.abc.StageAxis]:
        return self._axes

    def move_by(self, delta: typing.Mapping[str, float]) -> None:
        pass

    def move_to(self, position: typing.Mapping[str, float]) -> None:
        pass