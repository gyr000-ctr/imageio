# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Read/Write NIfTI-1 files.

This plugin implements native support for the NIfTI-1 format (``.nii`` and
gzip-compressed ``.nii.gz``), the de-facto standard for storing medical and
neuroimaging volumetric data. It is implemented in pure Python and only
depends on ``numpy``.

Layout
------
A NIfTI-1 file starts with a 348 byte header, followed by a 4 byte extension
flag and (optionally) extensions, and finally the raw voxel data. This plugin
parses the header fields needed to interpret that data:

- ``dim`` / ``datatype`` / ``bitpix`` describe the shape and voxel type,
- ``pixdim`` describes the voxel spacing,
- ``sform`` (an affine matrix) and ``qform`` (a quaternion plus offset)
  describe the mapping from voxel indices to world coordinates,
- ``scl_slope`` / ``scl_inter`` describe the linear transform between stored
  voxel values and physical values.

Reading
-------
:meth:`NiftiPlugin.read` returns the voxel data as a :class:`numpy.ndarray`
whose axes correspond to the first, second, ... spatial axis of the file
(nibabel's ``(i, j, k)`` order). This means the affine matrix maps array
indices directly to world coordinates. By default the
``scl_slope``/``scl_inter`` scaling is applied, which yields ``float64``; pass
``apply_scaling=False`` to obtain the raw stored values and their native
dtype.

The spatial information is exposed through
:meth:`NiftiPlugin.metadata` as ``affine`` (preferring ``sform`` over
``qform``, as the NIfTI-1 specification prescribes), plus ``sform`` and
``qform`` when present.

Writing
-------
:meth:`NiftiPlugin.write` stores a single ndimage with up to seven
dimensions. The affine is written to the ``sform`` fields and, for
axis-aligned affines, to the ``qform`` fields as well. A caller-provided
``scl_slope``/``scl_inter`` is stored unchanged, so spatial calibration and
value scaling can be preserved across a read/write round trip.

Errors
------
Malformed files raise specific exceptions:

- :class:`NiftiFormatError` for a bad ``magic`` or an incomplete header,
- :class:`TruncatedNiftiError` when the voxel data ends prematurely,
- :class:`UnknownNiftiDatatypeError` for an unsupported ``datatype`` code.

All three subclass both :class:`OSError` and :class:`ValueError`, so either
can be used to catch them.

Parameters
----------
affine : array_like
    A ``(4, 4)`` or ``(3, 4)`` affine mapping voxel indices to world
    coordinates. Used when writing; defaults to an identity matrix.
scl_slope, scl_inter : float
    Linear scaling between stored and physical values. Used when writing;
    default to ``1.0`` and ``0.0``.
pixdim : sequence of float
    Voxel spacing. Used when writing; defaults to the spacing encoded in
    ``affine``.
metadata : dict
    Additional header fields to write. See the module docstring of
    :func:`NiftiPlugin.write` for the supported keys.

Methods
-------
.. note::
    Check the respective function for a list of supported kwargs and detailed
    documentation.

.. autosummary::
    :toctree:

    NiftiPlugin.read
    NiftiPlugin.iter
    NiftiPlugin.write
    NiftiPlugin.properties
    NiftiPlugin.metadata

"""

import gzip
from io import BytesIO
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.request import URI_BYTES, InitializationError, IOMode, Request
from ..core.v3_plugin_api import ImageProperties, PluginV3
from ..typing import ArrayLike

# ``HEADER_DTYPE.itemsize`` is 348, the size of the NIfTI-1 header.
HEADER_DTYPE = np.dtype(
    [
        ("sizeof_hdr", "<i4"),
        ("data_type", "S10"),
        ("db_name", "S18"),
        ("extents", "<i4"),
        ("session_error", "<i2"),
        ("regular", "S1"),
        ("dim_info", "u1"),
        ("dim", "<i2", (8,)),
        ("intent_p1", "<f4"),
        ("intent_p2", "<f4"),
        ("intent_p3", "<f4"),
        ("intent_code", "<i2"),
        ("datatype", "<i2"),
        ("bitpix", "<i2"),
        ("slice_start", "<i2"),
        ("pixdim", "<f4", (8,)),
        ("vox_offset", "<f4"),
        ("scl_slope", "<f4"),
        ("scl_inter", "<f4"),
        ("slice_end", "<i2"),
        ("slice_code", "u1"),
        ("xyzt_units", "u1"),
        ("cal_max", "<f4"),
        ("cal_min", "<f4"),
        ("slice_duration", "<f4"),
        ("toffset", "<f4"),
        ("glmax", "<i4"),
        ("glmin", "<i4"),
        ("descrip", "S80"),
        ("aux_file", "S24"),
        ("qform_code", "<i2"),
        ("sform_code", "<i2"),
        ("quatern_b", "<f4"),
        ("quatern_c", "<f4"),
        ("quatern_d", "<f4"),
        ("qoffset_x", "<f4"),
        ("qoffset_y", "<f4"),
        ("qoffset_z", "<f4"),
        ("srow_x", "<f4", (4,)),
        ("srow_y", "<f4", (4,)),
        ("srow_z", "<f4", (4,)),
        ("intent_name", "S16"),
        ("magic", "S4"),
    ]
)

HEADER_SIZE = HEADER_DTYPE.itemsize  # 348
EXTENSION_FLAG_SIZE = 4
MIN_VOX_OFFSET = HEADER_SIZE + EXTENSION_FLAG_SIZE  # 352

# ``sizeof_hdr`` is 348 for NIfTI-1 and 540 for NIfTI-2.
NIFTI2_HEADER_SIZE = 540

MAGIC_SINGLE_FILE = b"n+1"
MAGIC_PAIR_FILE = b"ni1"

# NIfTI-1 datatype codes that map directly onto a numpy dtype.
DATATYPE_CODES = {
    2: np.dtype("<u1"),  # DT_UINT8
    4: np.dtype("<i2"),  # DT_INT16
    8: np.dtype("<i4"),  # DT_INT32
    16: np.dtype("<f4"),  # DT_FLOAT32
    64: np.dtype("<f8"),  # DT_FLOAT64
    256: np.dtype("<i1"),  # DT_INT8
    512: np.dtype("<u2"),  # DT_UINT16
    768: np.dtype("<u4"),  # DT_UINT32
    1024: np.dtype("<i8"),  # DT_INT64
    1280: np.dtype("<u8"),  # DT_UINT64
}

# Codes that describe packed color data; they do not map to a numpy dtype.
DATATYPE_PACKED_COLOR = {128, 2304}

# Coordinate system codes (NIfTI-1 spec). A value of 0 means "unknown".
XFORM_UNKNOWN = 0

# Header fields reported by ``metadata``. Fields already interpreted by
# ``read`` (shape, dtype, scaling) are handled separately.
METADATA_FIELDS = (
    "dim_info",
    "intent_code",
    "intent_p1",
    "intent_p2",
    "intent_p3",
    "intent_name",
    "slice_code",
    "slice_start",
    "slice_end",
    "slice_duration",
    "toffset",
    "xyzt_units",
    "cal_max",
    "cal_min",
    "descrip",
    "aux_file",
    "qform_code",
    "sform_code",
)

# Header fields accepted through the ``metadata`` kwarg of ``write``.
METADATA_WRITABLE = (
    "dim_info",
    "intent_code",
    "intent_p1",
    "intent_p2",
    "intent_p3",
    "intent_name",
    "slice_code",
    "slice_start",
    "slice_end",
    "slice_duration",
    "toffset",
    "xyzt_units",
    "cal_max",
    "cal_min",
    "descrip",
    "aux_file",
    "qform_code",
    "sform_code",
    "qfac",
    "pixdim",
    "scl_slope",
    "scl_inter",
)


class NiftiFormatError(OSError, ValueError):
    """The resource is not a structurally valid NIfTI-1 file.

    Raised for a bad ``magic``, an incomplete header, or an inconsistent
    header. It derives from both :class:`OSError` (ImageIO's usual error type
    for unreadable resources) and :class:`ValueError`, so either can be
    caught.
    """

    pass


class TruncatedNiftiError(NiftiFormatError):
    """The file ends before all voxel data announced in the header.

    The header was parsed successfully, but fewer than
    ``nvox * dtype.itemsize`` data bytes are available. The announced and
    available byte counts are available as :attr:`expected` and
    :attr:`actual`.
    """

    def __init__(self, message: str, *, expected: int, actual: int) -> None:
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class UnknownNiftiDatatypeError(NiftiFormatError):
    """The header declares a ``datatype`` this plugin cannot decode.

    The offending numeric code is available as :attr:`datatype`.
    """

    def __init__(self, message: str, *, datatype: int) -> None:
        super().__init__(message)
        self.datatype = datatype


def _has_nifti_content(file_obj) -> bool:
    """Cheaply sniff whether ``file_obj`` starts with NIfTI-1 data.

    The header's first field (``sizeof_hdr``) must equal 348 in the file's byte
    order, and ``magic`` (at offset 344) must be ``n+1`` or ``ni1``. Both are
    checked so that unrelated data is never mistaken for NIfTI. Gzip-compressed
    streams are not sniffed here (that would require decompression); they are
    only recognized by name.
    """
    position = file_obj.tell()
    try:
        file_obj.seek(0)
        signature = file_obj.read(2)
        if signature == b"\x1f\x8b":
            return False
        file_obj.seek(0)
        prefix = file_obj.read(HEADER_SIZE)
    finally:
        file_obj.seek(position)

    if len(prefix) < HEADER_SIZE:
        return False

    as_little = int.from_bytes(prefix[:4], "little", signed=True)
    as_big = int.from_bytes(prefix[:4], "big", signed=True)
    if as_little != HEADER_SIZE and as_big != HEADER_SIZE:
        return False

    return prefix[344:347] in (MAGIC_SINGLE_FILE, MAGIC_PAIR_FILE)


def _is_nifti_name(request: Request) -> bool:
    """Whether the request explicitly names a NIfTI file."""
    extension = request.extension
    if extension in (".nii", ".nii.gz"):
        return True

    filename = request.filename
    if isinstance(filename, str):
        lowered = filename.lower()
        if lowered.endswith(".nii") or lowered.endswith(".nii.gz"):
            return True

    return False


def _wants_gzip(request: Request) -> bool:
    """Whether the write target should be gzip-compressed."""
    if request.extension == ".nii.gz":
        return True
    filename = request.filename
    if isinstance(filename, str):
        return filename.lower().endswith(".nii.gz")
    return False


def _detect_byteorder(raw_header: bytes) -> str:
    """Determine the byte order of a NIfTI-1 header from ``sizeof_hdr``.

    ``sizeof_hdr`` is the first field and always equals 348, so its raw bytes
    unambiguously reveal the endianness. Raises for NIfTI-2 files, which this
    plugin does not support.
    """
    as_little = int.from_bytes(raw_header[:4], "little", signed=True)
    as_big = int.from_bytes(raw_header[:4], "big", signed=True)

    if as_little == HEADER_SIZE:
        return "<"
    if as_big == HEADER_SIZE:
        return ">"
    if NIFTI2_HEADER_SIZE in (as_little, as_big):
        raise NiftiFormatError(
            "This is a NIfTI-2 file (sizeof_hdr is 540). Only NIfTI-1 is "
            "supported; convert the file to NIfTI-1 first."
        )
    raise NiftiFormatError(
        "Invalid NIfTI-1 header: sizeof_hdr is "
        f"{as_little} (little endian) / {as_big} (big endian), but must be "
        f"{HEADER_SIZE}."
    )


def _parse_header(raw: bytes, byteorder: str = "<") -> Dict[str, Any]:
    """Decode the raw 348 byte header into a plain ``dict``."""
    dtype = HEADER_DTYPE.newbyteorder(byteorder)
    header = np.frombuffer(raw, dtype=dtype, count=1)[0]
    parsed: Dict[str, Any] = {}
    for name in HEADER_DTYPE.names:
        value = header[name]
        if isinstance(value, np.ndarray):
            parsed[name] = value.tolist()
        elif isinstance(value, np.generic):
            parsed[name] = value.item()
        else:  # pragma: no cover - defensive
            parsed[name] = value
    return parsed


def _collapse_shape(dim: Sequence[int]) -> Tuple[int, ...]:
    """Return the spatial shape described by ``dim``.

    Trailing singleton dimensions beyond ``dim[0]`` are not part of the
    declared rank, so they are not reported.
    """
    ndim = int(dim[0])
    if ndim < 1 or ndim > 7:
        raise NiftiFormatError(
            "Invalid NIfTI-1 header: dim[0] is "
            f"{ndim}, which is outside the valid range 1..7."
        )

    shape = tuple(int(size) for size in dim[1 : ndim + 1])
    if any(size < 1 for size in shape):
        raise NiftiFormatError(
            "Invalid NIfTI-1 header: dim declares a non-positive size " f"({shape})."
        )
    return shape


def _quaternion_to_affine(header: Mapping[str, Any]) -> np.ndarray:
    """Build a ``(4, 4)`` affine from the NIfTI-1 ``qform`` fields."""
    b = float(header["quatern_b"])
    c = float(header["quatern_c"])
    d = float(header["quatern_d"])

    a_squared = 1.0 - (b * b + c * c + d * d)
    if a_squared < 1e-7:
        # Slightly out-of-spec files store a non-normalized quaternion; the
        # NIfTI-1 spec recommends clamping to zero in this case.
        a = 0.0
    else:
        a = float(np.sqrt(a_squared))

    rotation = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - c * c - b * b],
        ],
        dtype=np.float64,
    )

    pixdim = np.asarray(header["pixdim"], dtype=np.float64)
    qfac = -1.0 if pixdim[0] < 0 else 1.0
    zooms = np.array([pixdim[1], pixdim[2], pixdim[3] * qfac], dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rotation * zooms[np.newaxis, :]
    affine[:3, 3] = (
        float(header["qoffset_x"]),
        float(header["qoffset_y"]),
        float(header["qoffset_z"]),
    )
    return affine


def _sform_to_affine(header: Mapping[str, Any]) -> np.ndarray:
    """Build a ``(4, 4)`` affine from the NIfTI-1 ``srow_*`` fields."""
    affine = np.eye(4, dtype=np.float64)
    affine[0, :] = header["srow_x"]
    affine[1, :] = header["srow_y"]
    affine[2, :] = header["srow_z"]
    return affine


def _affine_to_quaternion(affine: np.ndarray) -> Tuple[float, ...]:
    """Decompose an affine into NIfTI-1 ``qform`` fields.

    Returns ``(quatern_b, quatern_c, quatern_d, qoffset_x, qoffset_y,
    qoffset_z, qfac)``.

    The decomposition assumes the linear part is a (possibly flipped)
    rotation scaled along its columns, which is the case for axis-aligned
    images. General shears are only approximated here; they are always stored
    losslessly in the ``sform``.
    """
    linear = np.asarray(affine, dtype=np.float64)[:3, :3]
    offsets = np.asarray(affine, dtype=np.float64)[:3, 3]

    zooms = np.linalg.norm(linear, axis=0)
    zooms[zooms == 0] = 1.0
    rotation = linear / zooms[np.newaxis, :]

    qfac = 1.0
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0
        qfac = -1.0

    trace = np.trace(rotation)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        a = 0.25 * s
        b = (rotation[2, 1] - rotation[1, 2]) / s
        c = (rotation[0, 2] - rotation[2, 0]) / s
        d = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        a = (rotation[2, 1] - rotation[1, 2]) / s
        b = 0.25 * s
        c = (rotation[0, 1] + rotation[1, 0]) / s
        d = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        a = (rotation[0, 2] - rotation[2, 0]) / s
        b = (rotation[0, 1] + rotation[1, 0]) / s
        c = 0.25 * s
        d = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        a = (rotation[1, 0] - rotation[0, 1]) / s
        b = (rotation[0, 2] + rotation[2, 0]) / s
        c = (rotation[1, 2] + rotation[2, 1]) / s
        d = 0.25 * s

    # qform stores only (b, c, d); the sign of a is normalized away.
    if a < 0:
        b, c, d = -b, -c, -d

    return (
        float(b),
        float(c),
        float(d),
        float(offsets[0]),
        float(offsets[1]),
        float(offsets[2]),
        float(qfac),
    )


def _numpy_to_nifti_code(dtype: np.dtype) -> int:
    """Map a numpy dtype to its NIfTI-1 datatype code."""
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        dtype = np.dtype("u1")
    dtype = dtype.newbyteorder("<")
    for code, candidate in DATATYPE_CODES.items():
        if candidate == dtype:
            return code
    raise ValueError(
        f"NIfTI-1 cannot store ndimages of dtype `{dtype}`. Supported dtypes "
        "are int8, int16, int32, int64, uint8, uint16, uint32, uint64, "
        "float32 and float64."
    )


def _gzip_compress(payload: bytes) -> bytes:
    buffer = BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        stream.write(payload)
    return buffer.getvalue()


class NiftiPlugin(PluginV3):
    """Read/Write NIfTI-1 files.

    Parameters
    ----------
    request : iio.Request
        A request object representing the resource to be operated on.
    **kwargs : Any
        Ignored; exists for API compatibility.

    """

    def __init__(self, request: Request, **kwargs) -> None:
        super().__init__(request)

        self._header: Optional[Dict[str, Any]] = None
        self._dtype: Optional[np.dtype] = None
        self._shape: Optional[Tuple[int, ...]] = None
        self._affine: Optional[np.ndarray] = None
        self._scaling: Tuple[float, float] = (1.0, 0.0)
        self._data_offset = MIN_VOX_OFFSET
        self._compressed = False
        self._byteorder = "<"
        self._gzip_stream = None
        self._pending_error: Optional[NiftiFormatError] = None
        self._data_cache: Optional[bytes] = None

        if request.mode.io_mode == IOMode.read:
            self._open_for_read()
        else:
            self._check_write_target()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _check_write_target(self) -> None:
        """Reject write targets that are not NIfTI files."""
        if self.request._uri_type == URI_BYTES:
            return
        if _is_nifti_name(self.request):
            return
        raise InitializationError(
            "The NIFTI plugin cannot write "
            f"`{self.request.extension or self.request.filename}`; NIfTI-1 "
            "only supports `.nii` and `.nii.gz`."
        )

    def _open_for_read(self) -> None:
        strict = _is_nifti_name(self.request)
        if not strict:
            try:
                strict = _has_nifti_content(self.request.get_file())
            except Exception:  # pragma: no cover - unreadable resource
                strict = False

        try:
            self._parse_from_file()
        except NiftiFormatError as error:
            self._rewind_and_close()
            if strict:
                # Defer the error to the first read/metadata access so that
                # plugin discovery still succeeds (the extension says this is
                # ours) while the user gets a distinguishable error.
                self._pending_error = error
            else:
                # During discovery of an unrelated file, behave like every
                # other plugin and report that this is not our format.
                raise InitializationError(
                    "The resource does not contain a NIfTI-1 file."
                ) from error

    def _rewind_and_close(self) -> None:
        """Release the gzip reader and rewind the shared file object."""
        if self._gzip_stream is not None:
            self._gzip_stream.close()
            self._gzip_stream = None
        # Reading the header moved the file object; other plugins (during
        # fallback discovery) expect to start from the beginning.
        try:
            self.request.get_file().seek(0)
        except Exception:  # pragma: no cover - best-effort cleanup
            pass

    def _parse_from_file(self) -> None:
        file = self.request.get_file()
        file.seek(0)

        # Detect gzip compression from the stream signature rather than from
        # the filename so that files with the wrong suffix still work.
        signature = file.read(2)
        file.seek(0)

        source = file
        if signature == b"\x1f\x8b":
            self._compressed = True
            try:
                source = gzip.GzipFile(fileobj=file, mode="rb")
            except OSError as error:
                raise NiftiFormatError(
                    f"Could not read the gzip-compressed NIfTI-1 file: {error}"
                ) from error
            self._gzip_stream = source

        try:
            raw_header = source.read(HEADER_SIZE)
        except (OSError, EOFError) as error:
            raise NiftiFormatError(
                f"Could not decompress the NIfTI-1 header: {error}"
            ) from error

        if len(raw_header) < HEADER_SIZE:
            raise TruncatedNiftiError(
                "Truncated NIfTI-1 file: the header needs "
                f"{HEADER_SIZE} bytes, but only {len(raw_header)} are "
                "available.",
                expected=HEADER_SIZE,
                actual=len(raw_header),
            )

        self._byteorder = _detect_byteorder(raw_header)
        header = _parse_header(raw_header, self._byteorder)
        magic = bytes(header["magic"])[:3]
        if magic not in (MAGIC_SINGLE_FILE, MAGIC_PAIR_FILE):
            raise NiftiFormatError(
                "Bad NIfTI-1 magic: expected b'n+1' (single file) or b'ni1' "
                f"(header/image pair), found {bytes(header['magic'])!r}."
            )

        if magic == MAGIC_PAIR_FILE:
            raise NiftiFormatError(
                "This is a NIfTI-1 header/image pair (`magic` is b'ni1'). This "
                "plugin only reads single-file NIfTI; convert the pair to a "
                "`.nii` or `.nii.gz` file first."
            )

        self._header = header
        self._shape = _collapse_shape(header["dim"])
        self._dtype = self._resolve_dtype(header)

        sform_code = int(header["sform_code"])
        qform_code = int(header["qform_code"])
        if sform_code > XFORM_UNKNOWN:
            self._affine = _sform_to_affine(header)
        elif qform_code > XFORM_UNKNOWN:
            self._affine = _quaternion_to_affine(header)
        else:
            # No spatial transform stored; fall back to voxel spacing.
            self._affine = np.eye(4, dtype=np.float64)
            pixdim = np.asarray(header["pixdim"], dtype=np.float64)
            for axis in range(3):
                if pixdim[axis + 1] > 0:
                    self._affine[axis, axis] = pixdim[axis + 1]

        slope = float(header["scl_slope"])
        inter = float(header["scl_inter"])
        if not np.isfinite(slope) or slope == 0.0:
            # A zero or non-finite slope means "no scaling": both the slope
            # and the intercept are ignored (NIfTI-1 spec).
            slope = 1.0
            inter = 0.0
        elif not np.isfinite(inter):
            inter = 0.0
        self._scaling = (slope, inter)

        offset = int(float(header["vox_offset"]))
        if offset < MIN_VOX_OFFSET:
            # Some writers leave vox_offset at 0. A single-file NIfTI-1 image
            # must have its data past the header and extension flag.
            offset = MIN_VOX_OFFSET
        self._data_offset = offset

        self._validate_data_size()

    def _resolve_dtype(self, header: Mapping[str, Any]) -> np.dtype:
        datatype = int(header["datatype"])

        if datatype in DATATYPE_PACKED_COLOR:
            raise UnknownNiftiDatatypeError(
                "NIfTI-1 datatype "
                f"{datatype} stores packed RGB/RGBA voxels, which this plugin "
                "does not support.",
                datatype=datatype,
            )
        if datatype not in DATATYPE_CODES:
            raise UnknownNiftiDatatypeError(
                f"Unknown NIfTI-1 datatype code {datatype}.",
                datatype=datatype,
            )

        dtype = DATATYPE_CODES[datatype].newbyteorder(self._byteorder)
        bitpix = int(header["bitpix"])
        if bitpix not in (0, dtype.itemsize * 8):
            raise NiftiFormatError(
                "Inconsistent NIfTI-1 header: datatype "
                f"{datatype} implies {dtype.itemsize * 8} bit voxels, "
                f"but bitpix is {bitpix}."
            )
        return dtype

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def _ensure_ready(self) -> None:
        if self._pending_error is not None:
            raise self._pending_error

    def _data_byte_count(self) -> int:
        return int(np.prod(self._shape, dtype=np.int64)) * self._dtype.itemsize

    def _available_data_bytes(self) -> Optional[int]:
        """Number of decompressed data bytes left in the stream.

        Returns ``None`` if this cannot be determined without decompressing
        the rest of a gzip stream.
        """
        if self._compressed:
            return None

        file = self.request.get_file()
        position = file.tell()
        file.seek(0, 2)
        end = file.tell()
        file.seek(position)
        return max(end - self._data_offset, 0)

    def _validate_data_size(self) -> None:
        """Raise if the uncompressed file is shorter than its data."""
        expected = self._data_byte_count()
        available = self._available_data_bytes()
        if available is not None and available < expected:
            raise TruncatedNiftiError(
                "Truncated NIfTI-1 file: expected "
                f"{expected} bytes of voxel data (shape {self._shape}, dtype "
                f"{self._dtype}), but only {available} bytes are available.",
                expected=expected,
                actual=available,
            )

    def _read_data_bytes(self) -> bytes:
        if self._compressed:
            if self._data_cache is None:
                assert self._gzip_stream is not None
                try:
                    # The header was read sequentially, so this is the data
                    # starting right after it; advance to ``vox_offset``.
                    skip = self._data_offset - HEADER_SIZE
                    if skip > 0:
                        self._gzip_stream.read(skip)
                    payload = self._gzip_stream.read()
                except (OSError, EOFError) as error:
                    raise TruncatedNiftiError(
                        "Truncated gzip-compressed NIfTI-1 file: " f"{error}",
                        expected=self._data_byte_count(),
                        actual=0,
                    ) from error
                self._data_cache = payload
            else:
                payload = self._data_cache

            expected = self._data_byte_count()
            if len(payload) < expected:
                raise TruncatedNiftiError(
                    "Truncated NIfTI-1 file: expected "
                    f"{expected} bytes of voxel data after decompression, but "
                    f"only {len(payload)} bytes are available.",
                    expected=expected,
                    actual=len(payload),
                )
            return payload

        file = self.request.get_file()
        file.seek(self._data_offset)
        expected = self._data_byte_count()
        payload = file.read(expected)
        if len(payload) < expected:
            raise TruncatedNiftiError(
                "Truncated NIfTI-1 file: expected "
                f"{expected} bytes of voxel data (shape {self._shape}, dtype "
                f"{self._dtype}), but only {len(payload)} bytes are available.",
                expected=expected,
                actual=len(payload),
            )
        return payload

    def read(
        self, *, index: int = 0, apply_scaling: bool = True, **kwargs
    ) -> np.ndarray:
        """Read the voxel data of the NIfTI-1 file.

        Parameters
        ----------
        index : int
            NIfTI-1 files store a single ndimage, so only ``0`` (the default),
            ``None`` and ``...`` are accepted.
        apply_scaling : bool
            If True (default), apply ``scl_slope`` and ``scl_inter`` and
            return ``float64``. If False, return the raw stored values in
            their native dtype.
        **kwargs : Any
            Ignored; exists for API compatibility.

        Returns
        -------
        ndimage : np.ndarray
            The voxel data in file axis order.
        """
        self._ensure_ready()

        if index not in (0, None, Ellipsis):
            raise IndexError(
                f"Index `{index}` is out of bounds; a NIfTI-1 file stores a "
                "single ndimage."
            )

        payload = self._read_data_bytes()
        data = np.frombuffer(payload, dtype=self._dtype)
        data = data[: int(np.prod(self._shape, dtype=np.int64))]
        # NIfTI stores its first dimension as the fastest-varying one, so the
        # data is reshaped in Fortran order. The resulting array's axes then
        # match the affine's columns (nibabel's (i, j, k) convention).
        data = data.reshape(self._shape, order="F").astype(
            self._dtype.newbyteorder("="), copy=True
        )

        if apply_scaling:
            slope, inter = self._scaling
            if slope != 1.0:
                data = data.astype(np.float64) * slope
            if inter != 0.0:
                data = data.astype(np.float64) + inter

        return data

    def iter(self, **kwargs) -> Iterator[np.ndarray]:
        """Yield the single ndimage stored in the file."""
        yield self.read(**kwargs)

    def properties(self, index: int = 0, **kwargs) -> ImageProperties:
        """Standardized ndimage metadata.

        Parameters
        ----------
        index : int
            NIfTI-1 files store a single ndimage; only ``0`` (the default),
            ``None`` and ``...`` are accepted.

        Returns
        -------
        properties : ImageProperties
            Shape, dtype and voxel ``spacing`` of the ndimage. ``dtype``
            reflects what :meth:`read` returns: ``float64`` when non-trivial
            scaling is applied, the stored dtype otherwise.
        """
        self._ensure_ready()

        if index not in (0, None, Ellipsis):
            raise IndexError(
                f"Index `{index}` is out of bounds; a NIfTI-1 file stores a "
                "single ndimage."
            )

        slope, inter = self._scaling
        if slope != 1.0 or inter != 0.0:
            dtype = np.dtype(np.float64)
        else:
            dtype = self._dtype.newbyteorder("=")
        pixdim = np.abs(np.asarray(self._header["pixdim"][1:], dtype=np.float64))
        spacing = tuple(float(value) for value in pixdim[: len(self._shape)])

        return ImageProperties(
            shape=self._shape,
            dtype=dtype,
            spacing=spacing,
            is_batch=False,
        )

    def metadata(
        self, index: int = 0, exclude_applied: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """NIfTI-1 specific metadata.

        Parameters
        ----------
        index : int
            Ignored; NIfTI-1 files only store global metadata.
        exclude_applied : bool
            If True (default), omit ``scl_slope``/``scl_inter`` because
            :meth:`read` applies them. Set to False to inspect them.
        **kwargs : Any
            Ignored; exists for API compatibility.

        Returns
        -------
        metadata : dict
            The parsed header fields plus the derived ``affine`` matrix and,
            when present, the ``sform`` and ``qform`` matrices.
        """
        self._ensure_ready()
        header = self._header

        meta: Dict[str, Any] = {}
        for field in METADATA_FIELDS:
            value = header[field]
            if isinstance(value, bytes):
                value = value.rstrip(b"\x00").decode("latin1")
            meta[field] = value

        meta["dim"] = [int(value) for value in header["dim"]]
        meta["datatype"] = int(header["datatype"])
        meta["bitpix"] = int(header["bitpix"])
        meta["pixdim"] = [float(value) for value in header["pixdim"]]
        meta["affine"] = np.array(self._affine, dtype=np.float64, copy=True)

        if int(header["sform_code"]) > XFORM_UNKNOWN:
            meta["sform"] = _sform_to_affine(header)
        if int(header["qform_code"]) > XFORM_UNKNOWN:
            meta["qform"] = _quaternion_to_affine(header)

        if not exclude_applied:
            meta["scl_slope"] = float(header["scl_slope"])
            meta["scl_inter"] = float(header["scl_inter"])

        return meta

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------
    def write(
        self,
        ndimage: ArrayLike,
        *,
        affine: Optional[ArrayLike] = None,
        scl_slope: Optional[float] = None,
        scl_inter: Optional[float] = None,
        pixdim: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[bytes]:
        """Write a ndimage as a NIfTI-1 file.

        Parameters
        ----------
        ndimage : ArrayLike
            The ndimage to write. It must have between 1 and 7 dimensions and
            a dtype NIfTI-1 can store (int8/16/32/64, uint8/16/32/64,
            float32/64).
        affine : array_like
            ``(4, 4)`` or ``(3, 4)`` affine mapping voxel indices to world
            coordinates. Defaults to the identity matrix.
        scl_slope, scl_inter : float
            Linear scaling stored in the header. Default to ``1.0``/``0.0``
            (or the values given in ``metadata``).
        pixdim : sequence of float
            Voxel spacing along the three spatial axes. Defaults to the
            column norms of ``affine``.
        metadata : dict
            Additional header fields. Supported keys are ``dim_info``,
            ``intent_code``, ``intent_p1``, ``intent_p2``, ``intent_p3``,
            ``intent_name``, ``slice_code``, ``slice_start``, ``slice_end``,
            ``slice_duration``, ``toffset``, ``xyzt_units``, ``cal_max``,
            ``cal_min``, ``descrip``, ``aux_file``, ``qform_code``,
            ``sform_code``, ``qfac``, ``pixdim``, ``scl_slope`` and
            ``scl_inter``. Unknown keys raise ``ValueError``.
        **kwargs : Any
            Ignored; exists for API compatibility.

        Returns
        -------
        result : None or bytes
            ``None``, except when the target is the special ``"<bytes>"``
            resource, in which case the encoded NIfTI-1 file is returned.
        """
        data = np.asarray(ndimage)
        if data.ndim < 1 or data.ndim > 7:
            raise ValueError(
                "NIfTI-1 supports ndimages with between 1 and 7 dimensions, "
                f"but the given ndimage has {data.ndim} dimensions."
            )
        if data.dtype.byteorder == ">":
            data = data.astype(data.dtype.newbyteorder("<"))

        custom = dict(metadata) if metadata else {}
        unknown = set(custom) - set(METADATA_WRITABLE)
        if unknown:
            raise ValueError(
                "Unsupported NIfTI-1 metadata key(s): " + ", ".join(sorted(unknown))
            )

        affine_matrix = self._prepare_affine(affine, pixdim)

        if scl_slope is None:
            scl_slope = float(custom.get("scl_slope", 1.0))
        if scl_inter is None:
            scl_inter = float(custom.get("scl_inter", 0.0))

        header = self._build_header(
            data, affine_matrix, float(scl_slope), float(scl_inter), custom
        )
        voxels = np.ascontiguousarray(
            data.astype(data.dtype.newbyteorder("<"), copy=False)
        )
        # Reverse of the Fortran-order reshape used by ``read``: serialise the
        # array so that the first axis is the fastest-varying one.
        payload = header.tobytes() + b"\x00\x00\x00\x00" + voxels.tobytes(order="F")

        if self.request._uri_type == URI_BYTES:
            return _gzip_compress(payload) if _wants_gzip(self.request) else payload

        if _wants_gzip(self.request):
            payload = _gzip_compress(payload)

        self.request.get_file().write(payload)
        return None

    @staticmethod
    def _prepare_affine(
        affine: Optional[ArrayLike], pixdim: Optional[Sequence[float]]
    ) -> np.ndarray:
        if affine is None:
            affine_matrix = np.eye(4, dtype=np.float64)
        else:
            affine_matrix = np.asarray(affine, dtype=np.float64)
            if affine_matrix.shape == (3, 4):
                affine_matrix = np.vstack(
                    [affine_matrix, np.array([0.0, 0.0, 0.0, 1.0])]
                )
            if affine_matrix.shape != (4, 4):
                raise ValueError(
                    "`affine` must have shape (4, 4) or (3, 4), but has "
                    f"shape {affine_matrix.shape}."
                )

        if pixdim is not None:
            spacing = np.asarray(pixdim, dtype=np.float64).reshape(-1)
            if spacing.size < 3:
                raise ValueError("`pixdim` must contain at least three voxel spacings.")
            norms = np.linalg.norm(affine_matrix[:3, :3], axis=0)
            norms[norms == 0] = 1.0
            affine_matrix[:3, :3] = (
                affine_matrix[:3, :3] / norms[np.newaxis, :] * spacing[:3]
            )

        return affine_matrix

    @staticmethod
    def _build_header(
        data: np.ndarray,
        affine: np.ndarray,
        scl_slope: float,
        scl_inter: float,
        custom: Mapping[str, Any],
    ) -> np.ndarray:
        header = np.zeros(1, dtype=HEADER_DTYPE)

        header["sizeof_hdr"] = HEADER_SIZE
        header["magic"] = MAGIC_SINGLE_FILE
        header["regular"] = b"r"

        ndim = data.ndim
        dim = [ndim] + [int(size) for size in data.shape] + [1] * (7 - ndim)
        header["dim"] = np.asarray(dim, dtype=np.int16)

        header["datatype"] = _numpy_to_nifti_code(data.dtype)
        header["bitpix"] = data.dtype.itemsize * 8

        if custom.get("pixdim") is not None:
            spacing = np.asarray(custom["pixdim"], dtype=np.float64).reshape(-1)
            if spacing.size < 3:
                raise ValueError("`pixdim` must contain at least three voxel spacings.")
            spacing = spacing[:3].copy()
        else:
            spacing = np.linalg.norm(affine[:3, :3], axis=0)
            spacing[spacing == 0] = 1.0

        quatern_b, quatern_c, quatern_d, qx, qy, qz, qfac = _affine_to_quaternion(
            affine
        )
        if "qfac" in custom:
            qfac = -1.0 if float(custom["qfac"]) < 0 else 1.0

        pixdim = np.ones(8, dtype=np.float32)
        pixdim[0] = qfac
        pixdim[1:4] = spacing
        header["pixdim"] = pixdim

        header["vox_offset"] = float(MIN_VOX_OFFSET)
        header["scl_slope"] = float(scl_slope)
        header["scl_inter"] = float(scl_inter)

        header["srow_x"] = np.asarray(affine[0, :], dtype=np.float32)
        header["srow_y"] = np.asarray(affine[1, :], dtype=np.float32)
        header["srow_z"] = np.asarray(affine[2, :], dtype=np.float32)

        header["quatern_b"] = quatern_b
        header["quatern_c"] = quatern_c
        header["quatern_d"] = quatern_d
        header["qoffset_x"] = qx
        header["qoffset_y"] = qy
        header["qoffset_z"] = qz

        header["sform_code"] = int(custom.get("sform_code", 1))
        header["qform_code"] = int(custom.get("qform_code", 1))

        for field in METADATA_WRITABLE:
            if field in (
                "pixdim",
                "qfac",
                "sform_code",
                "qform_code",
                "scl_slope",
                "scl_inter",
            ):
                continue
            if field not in custom:
                continue
            value = custom[field]
            if isinstance(value, str):
                value = value.encode("latin1")
            header[field] = value

        return header

    def close(self) -> None:
        if self._gzip_stream is not None:
            # ``GzipFile`` does not close a caller-provided file object, but
            # closing it releases its internal buffers.
            self._gzip_stream.close()
            self._gzip_stream = None
        self.request.finish()
