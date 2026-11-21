# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read MINC1 files.

MINC1 is the first version of the Medical Imaging NetCDF file format. It
stores medical (volumetric) image data inside a NetCDF classic (CDF-1)
container. The files usually carry an ``.mnc`` extension and are optionally
compressed with gzip (``.mnc.gz``) or bzip2 (``.mnc.bz2``).

This plugin is read-only and implements a small NetCDF classic parser using
only the Python standard library, so it does not add any external
dependencies. It supports the uncompressed ``CDF\\x01`` magic as well as gzip
and bzip2 compressed variants.

The image data is reconstructed following the MINC1 conventions:

* The variable named ``image`` (or, if absent, the variable with the most
  dimensions) provides the voxel data.
* The array is returned in the order in which the dimensions are stored in
  the file. NetCDF stores the last dimension contiguously, i.e.
  ``(zspace, yspace, xspace)`` data comes back as a 3-D array of that shape.
* Integer data is scaled to floating point using the MINC1 conventions
  (``valid_range``/``image_min``/``image_max`` and the optional
  ``scale``/``offset`` or ``image_range`` attributes). Values outside of the
  valid range are missing values and become ``NaN``.

Distinguishable exceptions are raised for the different failure modes:

* :class:`MincFormatError` -- not a MINC1/NetCDF classic file, an unsupported
  NetCDF feature (e.g. record variables), or inconsistent dimensions.
* :class:`MincTypeError` -- the voxel type cannot be mapped to a numpy dtype.
* :class:`MincTruncatedError` -- the file ends before all declared data has
  been read.

All of these derive from :class:`MincError`, which in turn derives from
:class:`~imageio.core.request.InitializationError`, so the plugin selection
in :func:`imageio.v3.imopen` can fall back to another backend (e.g. ITK for
MINC2 files).

Methods
-------
.. note::
    Check the respective function for a list of supported kwargs and detailed
    documentation.

.. autosummary::
    :toctree:

    Minc1Plugin.read
    Minc1Plugin.iter
    Minc1Plugin.properties
    Minc1Plugin.metadata

"""

import bz2
import gzip
import struct
import zlib
from typing import Any, Dict, Iterator, Tuple

import numpy as np

from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties

# NetCDF numeric type codes (all NetCDF data is big-endian)
_NC_BYTE = 1
_NC_CHAR = 2
_NC_SHORT = 3
_NC_INT = 4
_NC_FLOAT = 5
_NC_DOUBLE = 6

_NC_DTYPES = {
    _NC_BYTE: np.dtype(">i1"),  # signed 1-byte integer
    _NC_CHAR: np.dtype("|S1"),  # 1-byte characters
    _NC_SHORT: np.dtype(">i2"),
    _NC_INT: np.dtype(">i4"),
    _NC_FLOAT: np.dtype(">f4"),
    _NC_DOUBLE: np.dtype(">f8"),
}

# NetCDF tag bytes
_NC_DIMENSION = 0x0A
_NC_VARIABLE = 0x0B
_NC_ATTRIBUTE = 0x0C

# Magic numbers used to detect compression
_GZIP_MAGIC = b"\x1f\x8b"
_BZIP2_MAGIC = b"BZ"


class MincError(InitializationError):
    """Base class for all errors raised by the MINC1 plugin."""


class MincFormatError(MincError):
    """Raised when a file is not valid MINC1/NetCDF or uses unsupported
    features, or when its dimensions are inconsistent."""


class MincTypeError(MincError):
    """Raised when the voxel type of a file cannot be mapped to a numpy
    dtype."""


class MincTruncatedError(MincError):
    """Raised when a file ends before all declared data could be read."""


class _NetCDFReader:
    """Minimal parser for the NetCDF classic (CDF-1) file format.

    The parser reads the complete header into python objects and remembers
    the (4-byte aligned) offsets of the non-record variables, so that their
    data is only read on demand. Only the subset of the format that is
    needed for MINC1 files is supported.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        self.dimensions: Dict[str, int] = {}
        self.attributes: Dict[str, Dict[str, Any]] = {}
        self.variables: Dict[str, Dict[str, Any]] = {}
        self._parse_header()

    # -- low level primitives -------------------------------------------------

    def _read_exact(self, count: int) -> bytes:
        """Read exactly ``count`` bytes or raise a truncation error."""

        data = self._stream.read(count)
        if len(data) != count:
            raise MincTruncatedError(
                f"truncated MINC1 file: expected {count} more bytes but got "
                f"{len(data)}."
            )
        return data

    def _read_u32(self) -> int:
        return struct.unpack(">I", self._read_exact(4))[0]

    def _read_tag(self) -> int:
        # Tags are padded to 4 bytes with zero bytes.
        tag = self._read_u32()
        return tag & 0xFF

    def _read_name(self) -> str:
        length = self._read_u32()
        raw = self._read_exact(length)
        # strings are padded to a multiple of 4 bytes
        pad = (-length) % 4
        if pad:
            self._read_exact(pad)
        return raw.decode("utf-8")

    def _skip_padding(self, count: int) -> None:
        pad = (-count) % 4
        if pad:
            self._read_exact(pad)

    # -- header parsing --------------------------------------------------------

    def _parse_header(self) -> None:
        magic = self._read_exact(4)
        if magic != b"CDF\x01":
            if magic[:3] == b"CDF":
                raise MincFormatError(
                    "unsupported NetCDF version: only the classic CDF-1 "
                    "format used by MINC1 is supported."
                )
            raise MincFormatError(
                "not a NetCDF classic (MINC1) file: missing the 'CDF\\x01' "
                "magic sequence."
            )

        numrecs = self._read_u32()
        if numrecs != 0:
            raise MincFormatError(
                "NetCDF files with a record (unlimited) dimension are not "
                "supported; MINC1 image files do not use them."
            )

        # dimension list
        tag = self._read_tag()
        if tag != _NC_DIMENSION:
            raise MincFormatError(
                f"malformed NetCDF header: expected dimension tag (0x0A), "
                f"got 0x{tag:02X}."
            )
        for _ in range(self._read_u32()):
            name = self._read_name()
            length = self._read_u32()
            self.dimensions[name] = length

        # global attributes
        tag = self._read_tag()
        if tag == _NC_ATTRIBUTE:
            self.attributes[""] = self._read_attribute_list(self._read_u32())
            tag = self._read_tag()

        # variable list (its attributes are part of each variable entry)
        if tag != _NC_VARIABLE:
            raise MincFormatError(
                f"malformed NetCDF header: expected variable tag (0x0B), "
                f"got 0x{tag:02X}."
            )
        for _ in range(self._read_u32()):
            self._read_variable()

        # The header ends here. All non-record variable data follows
        # back-to-back, each block padded to a multiple of 4 bytes.
        data_start = self._stream.tell()
        offset = data_start
        for name in self.variables:
            self.variables[name]["offset"] = offset
            size = self.variables[name]["size"]
            offset += size + ((-size) % 4)

    def _read_attribute_list(self, count: int) -> Dict[str, Any]:
        """Read ``count`` NetCDF attribute entries.

        Both the global attribute section and each variable entry store their
        attributes as a count followed by the attribute entries.
        """

        attrs: Dict[str, Any] = {}
        for _ in range(count):
            name = self._read_name()
            nc_type = self._read_u32()
            length = self._read_u32()
            dtype = _NC_DTYPES.get(nc_type)
            if dtype is None:
                raise MincTypeError(
                    f"unknown NetCDF attribute type code {nc_type} for "
                    f"attribute '{name}'."
                )
            raw = self._read_exact(dtype.itemsize * length)
            self._skip_padding(dtype.itemsize * length)
            values = np.frombuffer(raw, dtype=dtype, count=length)
            if nc_type == _NC_CHAR:
                value: Any = values.tobytes().rstrip(b"\x00").decode("utf-8")
            elif length == 1:
                value = values[0].item()
            else:
                value = values
            attrs[name] = value
        return attrs

    def _read_variable(self) -> None:
        name = self._read_name()
        dim_ids = [self._read_u32() for _ in range(self._read_u32())]
        # the attribute list is embedded in the variable entry
        attrs = self._read_attribute_list(self._read_u32())
        nc_type = self._read_u32()
        size = self._read_u32()  # variable data size in bytes

        dtype = _NC_DTYPES.get(nc_type)
        if dtype is None:
            raise MincTypeError(
                f"unknown NetCDF voxel type code {nc_type} for variable " f"'{name}'."
            )

        # Offsets are computed after the whole header has been parsed, since
        # all variable data is stored behind the header.
        self.variables[name] = {
            "dtype": dtype,
            "nc_type": nc_type,
            "dim_ids": dim_ids,
            "attrs": attrs,
            "offset": None,
            "size": size,
        }

    # -- data access -----------------------------------------------------------

    def read_variable(self, name: str) -> np.ndarray:
        """Return the raw (unscaled) data of variable ``name``."""

        info = self.variables[name]
        dim_names = list(self.dimensions)
        try:
            shape = tuple(self.dimensions[dim_names[i]] for i in info["dim_ids"])
        except IndexError as err:
            raise MincFormatError(
                f"dimension mismatch: variable '{name}' references a "
                f"dimension that does not exist in the file."
            ) from err

        count = int(np.prod(shape)) if shape else 1
        self._stream.seek(info["offset"])
        raw = self._read_exact(info["dtype"].itemsize * count)
        try:
            return np.frombuffer(raw, dtype=info["dtype"], count=count).reshape(shape)
        except ValueError as err:
            raise MincFormatError(
                f"dimension mismatch: variable '{name}' declares shape {shape} "
                f"but its data cannot be reshaped accordingly."
            ) from err


def _open_stream(file_object):
    """Detect compression and return a seekable binary stream.

    Detection is based on the content (magic bytes), so compressed files
    without a matching extension are handled as well. The underlying object
    is always seekable (a regular file, an in-memory buffer or a seekable
    network wrapper), so it can simply be rewound after the magic check.
    """

    magic = file_object.read(4)
    file_object.seek(0)

    if magic[:2] == _GZIP_MAGIC:
        return gzip.GzipFile(fileobj=file_object, mode="rb")
    if magic[:2] == _BZIP2_MAGIC:
        return bz2.BZ2File(file_object, mode="rb")

    return file_object


def _native_range(dtype: np.dtype) -> Tuple[float, float]:
    """Default MINC1 valid range for an integer storage dtype."""

    info = np.iinfo(dtype)
    return float(info.min + 1), float(info.max)


class Minc1Plugin(PluginV3):
    """Read MINC1 (NetCDF classic) medical image volumes."""

    def __init__(self, request: Request) -> None:
        """Instantiate a new MINC1 plugin object.

        Parameters
        ----------
        request : Request
            A request object representing the resource to be operated on.

        Raises
        ------
        MincFormatError
            If the resource is not a MINC1/NetCDF classic file or contains
            unsupported NetCDF features.
        MincTypeError
            If the voxel type of the image variable is unknown.
        MincTruncatedError
            If the header is truncated.
        """

        super().__init__(request)
        if request.mode.io_mode == IOMode.write:
            raise InitializationError("MINC1 files cannot be written.")

        self._file = self.request.get_file()
        stream = _open_stream(self._file)

        try:
            self._nc = _NetCDFReader(stream)
        except MincError:
            # already a distinguishable MINC1 error (format/type/truncation)
            raise
        except EOFError as err:
            # gzip/bz2 signal a truncated payload with EOFError
            raise MincTruncatedError(f"truncated MINC1 file: {err}") from err
        except (UnicodeDecodeError, struct.error, OSError, zlib.error) as err:
            # malformed header or a corrupt (non-truncated) compressed stream
            raise MincFormatError(
                f"MINC1 plugin cannot read the provided file: {err}"
            ) from err

        self._image_name = self._select_image_variable()
        info = self._nc.variables[self._image_name]
        self._dtype = info["dtype"]
        dim_names = list(self._nc.dimensions)
        try:
            self._shape = tuple(
                self._nc.dimensions[dim_names[dim_id]] for dim_id in info["dim_ids"]
            )
        except IndexError as err:
            raise MincFormatError(
                f"dimension mismatch: image variable '{self._image_name}' "
                f"references a dimension that does not exist in the file "
                f"(dimension ids {info['dim_ids']}, available: "
                f"{list(range(len(dim_names)))})."
            ) from err
        self._attrs = info["attrs"]
        self._scaled = self._dtype.kind in "iu"

    def _select_image_variable(self) -> str:
        """Find the variable holding the voxel data."""

        if "image" in self._nc.variables:
            return "image"

        candidates = [
            name
            for name, info in self._nc.variables.items()
            if info["nc_type"] != _NC_CHAR and info["dim_ids"]
        ]
        if not candidates:
            raise MincFormatError(
                "the file does not contain an 'image' variable or any other "
                "multi-dimensional variable with voxel data."
            )
        # fall back to the variable spanning the most dimensions
        return max(candidates, key=lambda n: len(self._nc.variables[n]["dim_ids"]))

    # -- scaling ---------------------------------------------------------------

    def _scaling_parameters(
        self,
    ) -> Tuple[float, float, Tuple[float, float]]:
        """Compute (slope, intercept, valid_range) from the MINC1 attributes.

        Two conventions exist:

        * the modern one uses ``valid_range`` together with optional
          ``scale``/``offset``;
        * the legacy one uses ``image_range``/``image_min``/``image_max``.

        The returned ``valid_range`` is always expressed in storage units and
        marks the stored integer values that are not MINC fill/missing data.
        It defaults to the native integer range with one extreme reserved as
        the fill value when the file does not declare one.
        """

        attrs = self._attrs

        if "valid_range" in attrs:
            valid = attrs["valid_range"]
            valid_min, valid_max = float(valid[0]), float(valid[1])
        else:
            # the MINC default valid range reserves one extreme for fill data
            valid_min, valid_max = _native_range(self._dtype)

        if "image_range" in attrs:
            # legacy convention: stored image_range maps onto image_min/max
            img_range = attrs["image_range"]
            range_min, range_max = float(img_range[0]), float(img_range[1])
            img_min = float(attrs["image_min"])
            img_max = float(attrs["image_max"])
            slope = (img_max - img_min) / (range_max - range_min)
            intercept = img_min - slope * range_min
        elif "image_min" in attrs and "image_max" in attrs:
            # image extents correspond to the (declared or default) valid range
            img_min = float(attrs["image_min"])
            img_max = float(attrs["image_max"])
            slope = (img_max - img_min) / (valid_max - valid_min)
            intercept = img_min - slope * valid_min
        elif "scale" in attrs or "offset" in attrs:
            slope = float(attrs.get("scale", 1.0))
            intercept = float(attrs.get("offset", 0.0))
        else:
            # no scaling information: return the stored values unchanged
            slope = 1.0
            intercept = 0.0

        return slope, intercept, (valid_min, valid_max)

    def _read_raw(self) -> np.ndarray:
        try:
            data = self._nc.read_variable(self._image_name)
        except MincError:
            # already a distinguishable MINC1 error (format/type/truncation)
            raise
        except EOFError as err:
            # truncated compressed payload encountered during data reads
            raise MincTruncatedError(
                f"truncated MINC1 file while reading '{self._image_name}': {err}"
            ) from err
        except (OSError, zlib.error) as err:
            # a corrupt (non-truncated) compressed stream
            raise MincFormatError(
                f"MINC1 file cannot be read while accessing "
                f"'{self._image_name}': {err}"
            ) from err

        expected = int(np.prod(self._shape)) if self._shape else 1
        if data.size != expected:
            raise MincFormatError(
                f"dimension mismatch: variable '{self._image_name}' holds "
                f"{data.size} values but its dimensions {self._shape} "
                f"require {expected}."
            )
        return data

    def _apply_scaling(self, data: np.ndarray) -> np.ndarray:
        if not self._scaled:
            return data

        slope, intercept, (valid_min, valid_max) = self._scaling_parameters()
        stored = data.astype(np.float64)
        out = stored * slope + intercept

        # Stored integers outside of the valid range are MINC fill/missing
        # values; the range is expressed in storage units.
        missing = (stored < valid_min) | (stored > valid_max)
        out[missing] = np.nan
        return out

    # -- public API ------------------------------------------------------------

    def read(self, *, index: int = ...) -> np.ndarray:
        """Read the image volume stored in the file.

        Parameters
        ----------
        index : int
            A MINC1 file holds a single volume. ``...`` (or ``None``) reads
            the whole volume. Integer indices other than ``0`` raise an
            ``IndexError``.

        Returns
        -------
        ndimage : np.ndarray
            The voxel data. Integer storage types are returned as
            ``float64`` after applying the MINC1 scaling; floating point
            storage types are returned unchanged. The axis order matches the
            dimension order stored in the file (typically
            ``(zspace, yspace, xspace)``).
        """

        if index is Ellipsis or index is None or index == 0:
            return self._apply_scaling(self._read_raw())
        if isinstance(index, int):
            raise IndexError(
                f"index {index} is out of range; a MINC1 file contains a "
                "single volume."
            )
        raise ValueError(f"invalid index {index!r}; use 0, None or ... .")

    def iter(self) -> Iterator[np.ndarray]:
        """Iterate over the images in the file.

        Yields
        ------
        ndimage : np.ndarray
            The single image volume stored in the file.
        """

        yield self.read(index=0)

    def properties(self, index: int = ...) -> ImageProperties:
        """Standardized ndimage metadata.

        Parameters
        ----------
        index : int
            Ignored. MINC1 files contain a single volume.

        Returns
        -------
        properties : ImageProperties
            A dataclass filled with standardized image metadata.
        """

        return ImageProperties(
            shape=self._shape,
            dtype=np.dtype(np.float64) if self._scaled else self._dtype,
            n_images=1,
            is_batch=False,
        )

    def metadata(
        self, index: int = ..., exclude_applied: bool = True
    ) -> Dict[str, Any]:
        """Format-specific MINC1 metadata.

        Parameters
        ----------
        index : int
            Ignored as MINC1 files only store global metadata.
        exclude_applied : bool
            If ``True`` (default), scaling attributes that have already been
            applied to the returned data are omitted.

        Returns
        -------
        metadata : dict
            The NetCDF dimensions, the image variable's attributes and the
            global attributes of the file.
        """

        meta: Dict[str, Any] = {
            "dimensions": dict(self._nc.dimensions),
            "image_variable": self._image_name,
            "voxel_type": int(self._nc.variables[self._image_name]["nc_type"]),
            "scaled": self._scaled,
        }
        meta.update(
            {
                f"global:{name}": value
                for name, value in self._nc.attributes.get("", {}).items()
            }
        )

        applied = {
            "valid_range",
            "image_min",
            "image_max",
            "image_range",
            "scale",
            "offset",
        }
        for name, value in self._attrs.items():
            if exclude_applied and self._scaled and name in applied:
                continue
            meta[name] = value
        return meta
