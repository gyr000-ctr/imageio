# -*- coding: utf-8 -*-
# imageio is distributed under the terms of the (new) BSD License.

"""Tests for the read-only MINC1 (NetCDF classic) plugin.

The tests synthesize MINC1/NetCDF classic files with a small writer
implemented in the test module, so no external binaries or test image
repositories are required.
"""

import bz2
import gzip
import struct

import numpy as np
import pytest

import imageio.config as config
import imageio.v3 as iio
from imageio.core.request import InitializationError
from imageio.plugins import minc
from imageio.plugins.minc import (
    Minc1Plugin,
    MincError,
    MincFormatError,
    MincTruncatedError,
    MincTypeError,
)

# ---------------------------------------------------------------------------
# Minimal NetCDF classic (CDF-1) writer -- enough to build MINC1-like files
# ---------------------------------------------------------------------------

NC_BYTE, NC_CHAR, NC_SHORT, NC_INT, NC_FLOAT, NC_DOUBLE = 1, 2, 3, 4, 5, 6


class NetCDFWriter:
    """Write NetCDF classic files without any external dependency."""

    def __init__(self):
        self._dimensions = []  # (name, length)
        self._globals = []  # (name, nc_type, numpy array / str)
        self._variables = []  # (name, dim_names, attrs, nc_type, array)

    def add_dimension(self, name, length):
        self._dimensions.append((name, length))
        return self

    def add_global(self, name, nc_type, values):
        self._globals.append((name, nc_type, values))
        return self

    def add_variable(self, name, dim_names, attrs, nc_type, array):
        self._variables.append(
            (name, list(dim_names), list(attrs), nc_type, np.asarray(array))
        )
        return self

    @staticmethod
    def _padded(raw):
        return raw + b"\x00" * ((-len(raw)) % 4)

    def _name(self, name):
        raw = name.encode("utf-8")
        return struct.pack(">I", len(raw)) + self._padded(raw)

    def _attr_value(self, nc_type, values):
        if nc_type == NC_CHAR:
            raw = np.asarray(values, dtype="S").tobytes()
            return raw, len(raw)
        array = np.asarray(values).astype(minc._NC_DTYPES[nc_type])
        return array.tobytes("C"), array.size

    def _attrs(self, attrs):
        out = struct.pack(">I", len(attrs))
        for name, nc_type, values in attrs:
            raw, count = self._attr_value(nc_type, values)
            out += self._name(name)
            out += struct.pack(">II", nc_type, count)
            out += self._padded(raw)
        return out

    def getvalue(self):
        buf = b"CDF\x01"
        buf += struct.pack(">I", 0)  # numrecs = 0 (no record dimension)

        # dimensions
        buf += struct.pack(">II", 0x0A, len(self._dimensions))
        for name, length in self._dimensions:
            buf += self._name(name) + struct.pack(">I", length)

        # global attributes (only emit the section when there are any)
        if self._globals:
            buf += struct.pack(">I", 0x0C) + self._attrs(self._globals)

        # variables
        buf += struct.pack(">II", 0x0B, len(self._variables))
        dim_index = {name: i for i, (name, _) in enumerate(self._dimensions)}
        data_section = b""
        for name, dim_names, attrs, nc_type, array in self._variables:
            dtype = minc._NC_DTYPES.get(nc_type, np.dtype(">i2"))
            buf += self._name(name)
            buf += struct.pack(">I", len(dim_names))
            buf += b"".join(struct.pack(">I", dim_index[n]) for n in dim_names)
            buf += self._attrs(attrs)
            if nc_type == NC_CHAR:
                raw, _ = self._attr_value(nc_type, array)
            else:
                raw = array.astype(dtype).tobytes("C")
            buf += struct.pack(">II", nc_type, len(raw))
            data_section += self._padded(raw)

        return buf + data_section


def _make_minc(
    array,
    dims=("zspace", "yspace", "xspace"),
    nc_type=NC_SHORT,
    attrs=(),
    globals_=(),
    image_name="image",
    extra_variables=(),
):
    """Build the bytes of a MINC1-like NetCDF classic file."""

    writer = NetCDFWriter()
    for name, length in zip(dims, np.asarray(array).shape):
        writer.add_dimension(name, length)
    for name, nc_t, values in globals_:
        writer.add_global(name, nc_t, values)
    writer.add_variable(image_name, dims, attrs, nc_type, array)
    for name, var_dims, var_attrs, var_type, var_array in extra_variables:
        writer.add_variable(name, var_dims, var_attrs, var_type, var_array)
    return writer.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_volume(tmp_path):
    """A small integer volume with explicit valid range and scale/offset."""

    data = np.arange(2 * 3 * 4, dtype=">i2").reshape(2, 3, 4) - 12
    attrs = (
        ("valid_range", NC_SHORT, np.array([-12, 11], dtype=">i2")),
        ("scale", NC_DOUBLE, np.array([0.5], dtype=">f8")),
        ("offset", NC_DOUBLE, np.array([1.0], dtype=">f8")),
    )
    path = tmp_path / "volume.mnc"
    path.write_bytes(_make_minc(data, attrs=attrs))
    return path, data


def _open_explicit(path):
    """Open with the plugin pinned by name.

    ``imopen`` wraps InitializationErrors raised by an explicitly selected
    plugin into OSError; the original, distinguishable MINC error remains
    available as ``__cause__``.
    """

    with pytest.raises(OSError) as exc_info:
        iio.imopen(path, "r", plugin="MINC1")
    return exc_info.value.__cause__


# ---------------------------------------------------------------------------
# Registration / open behavior
# ---------------------------------------------------------------------------


def test_plugin_registration():
    assert "MINC1" in config.known_plugins
    assert config.known_plugins["MINC1"].class_name == "Minc1Plugin"
    # MINC1 takes precedence over ITK for .mnc, but ITK remains a fallback
    assert config.known_extensions[".mnc"][0].priority == ["MINC1", "ITK"]
    assert config.known_extensions[".mnc2"][0].priority == ["ITK"]


def test_imopen(simple_volume):
    with iio.imopen(simple_volume[0], "r") as img:
        assert isinstance(img, Minc1Plugin)

    # explicit plugin selection works as well
    with iio.imopen(simple_volume[0], "r", plugin="MINC1") as img:
        assert isinstance(img, Minc1Plugin)


def test_read_only(tmp_path):
    with pytest.raises(OSError):
        iio.imopen(tmp_path / "new.mnc", "w", plugin="MINC1")


def test_not_a_minc_file(tmp_path):
    stub = tmp_path / "stub.mnc"
    stub.write_bytes(b"this is definitely not a NetCDF file")
    assert isinstance(_open_explicit(stub), MincFormatError)

    # automatic selection must not blow up, but report that no backend works
    with pytest.raises(OSError):
        iio.imopen(stub, "r")


def test_minc2_hdf5_is_rejected(tmp_path):
    # MINC2 files are HDF5 containers starting with the HDF5 signature
    path = tmp_path / "volume.mnc2"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 64)
    assert isinstance(_open_explicit(path), MincFormatError)


def test_unsupported_netcdf_version(tmp_path):
    path = tmp_path / "v2.mnc"
    path.write_bytes(b"CDF\x02" + b"\x00" * 32)
    assert isinstance(_open_explicit(path), MincFormatError)


# ---------------------------------------------------------------------------
# Reading: dimension order, dtypes and scaling
# ---------------------------------------------------------------------------


def test_dimension_order(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(-50, 50, size=(2, 3, 4)).astype(">i2")
    path = tmp_path / "order.mnc"
    path.write_bytes(_make_minc(data))

    out = iio.imread(path)
    assert out.shape == data.shape
    # without scale/offset the data is mapped 1:1 (as float64)
    np.testing.assert_allclose(out, data.astype(np.float64))


def test_2d_and_4d_volumes(tmp_path):
    for shape in ((4, 5), (2, 3, 4, 5)):
        data = np.arange(np.prod(shape), dtype=">i4").reshape(shape)
        path = tmp_path / f"vol_{len(shape)}d.mnc"
        path.write_bytes(
            _make_minc(
                data,
                dims=tuple(f"d{i}" for i in range(len(shape))),
                nc_type=NC_INT,
            )
        )
        out = iio.imread(path)
        assert out.shape == shape
        np.testing.assert_allclose(out, data.astype(np.float64))


def test_scale_and_offset(simple_volume):
    path, data = simple_volume
    out = iio.imread(path)
    expected = data.astype(np.float64) * 0.5 + 1.0
    np.testing.assert_allclose(out, expected)


def test_missing_values_become_nan(tmp_path):
    data = np.array([[0, 1, -32768, 32767, 5]], dtype=">i2")
    attrs = (("valid_range", NC_SHORT, np.array([0, 100], dtype=">i2")),)
    path = tmp_path / "missing.mnc"
    path.write_bytes(_make_minc(data, dims=("yspace", "xspace"), attrs=attrs))

    out = iio.imread(path)
    assert out.dtype == np.float64
    assert np.isnan(out[0, 2])
    assert np.isnan(out[0, 3])
    assert np.isfinite(out[0, 0]) and np.isfinite(out[0, 4])


def test_legacy_image_min_max_convention(tmp_path):
    # image extents -1..1 correspond to the default valid range -127..127
    data = np.array([0, 1, 127, -127], dtype=">i1")  # NC_BYTE
    attrs = (
        ("image_min", NC_DOUBLE, np.array([-1.0], dtype=">f8")),
        ("image_max", NC_DOUBLE, np.array([1.0], dtype=">f8")),
    )
    path = tmp_path / "legacy.mnc"
    path.write_bytes(_make_minc(data, dims=("xspace",), nc_type=NC_BYTE, attrs=attrs))
    out = iio.imread(path)
    np.testing.assert_allclose(out, data.astype(np.float64) / 127.0)


def test_legacy_image_range_convention(tmp_path):
    data = np.array([0, 127, -128], dtype=">i1")
    attrs = (
        ("image_range", NC_DOUBLE, np.array([-128.0, 127.0], dtype=">f8")),
        ("image_min", NC_DOUBLE, np.array([-1.0], dtype=">f8")),
        ("image_max", NC_DOUBLE, np.array([1.0], dtype=">f8")),
    )
    path = tmp_path / "image_range.mnc"
    path.write_bytes(_make_minc(data, dims=("xspace",), nc_type=NC_BYTE, attrs=attrs))
    out = iio.imread(path)
    # slope = 2/255, intercept = 1/255; -128 is outside the valid range
    np.testing.assert_allclose(out[:2], [1.0 / 255.0, 1.0])
    assert np.isnan(out[2])


def test_float_storage_is_not_scaled(tmp_path):
    data = np.linspace(-1.0, 1.0, 12, dtype=">f4").reshape(3, 4)
    path = tmp_path / "float.mnc"
    path.write_bytes(_make_minc(data, dims=("yspace", "xspace"), nc_type=NC_FLOAT))

    out = iio.imread(path)
    assert out.dtype == np.dtype(">f4")
    np.testing.assert_allclose(out, data)


def test_double_storage(tmp_path):
    data = np.linspace(0.0, 2.0, 6, dtype=">f8")
    path = tmp_path / "double.mnc"
    path.write_bytes(_make_minc(data, dims=("xspace",), nc_type=NC_DOUBLE))
    out = iio.imread(path)
    assert out.dtype == np.dtype(">f8")
    np.testing.assert_allclose(out, data)


def test_fallback_image_variable_selection(tmp_path):
    # no variable named "image": the largest multi-dimensional variable wins
    data = np.arange(6, dtype=">i2").reshape(2, 3)
    payload = _make_minc(
        data,
        dims=("yspace", "xspace"),
        image_name="mri",
        extra_variables=(("xspace", ("xspace",), (), NC_DOUBLE, np.arange(3.0)),),
    )
    path = tmp_path / "noimage.mnc"
    path.write_bytes(payload)
    out = iio.imread(path)
    np.testing.assert_allclose(out, data.astype(np.float64))


def test_iter_and_index(simple_volume):
    path, data = simple_volume
    with iio.imopen(path, "r") as img:
        frames = list(img.iter())
    assert len(frames) == 1
    np.testing.assert_allclose(frames[0], data.astype(np.float64) * 0.5 + 1.0)

    with iio.imopen(path, "r") as img:
        np.testing.assert_allclose(img.read(index=0), img.read(index=...))
        with pytest.raises(IndexError):
            img.read(index=1)


def test_properties_and_metadata(simple_volume):
    path, data = simple_volume

    props = iio.improps(path)
    assert props.shape == data.shape
    assert props.dtype == np.dtype(np.float64)
    assert props.n_images == 1
    assert not props.is_batch

    meta = iio.immeta(path)
    assert meta["dimensions"] == {"zspace": 2, "yspace": 3, "xspace": 4}
    assert meta["image_variable"] == "image"
    assert meta["voxel_type"] == NC_SHORT
    assert meta["scaled"] is True
    # applied scaling attributes are excluded by default
    assert "valid_range" not in meta and "scale" not in meta

    meta_full = iio.immeta(path, exclude_applied=False)
    assert meta_full["scale"] == pytest.approx(0.5)
    assert meta_full["offset"] == pytest.approx(1.0)


def test_global_attributes(tmp_path):
    data = np.ones((2, 2), dtype=">i2")
    payload = _make_minc(
        data,
        dims=("yspace", "xspace"),
        globals_=(("ident", NC_CHAR, "MINC test file"),),
    )
    path = tmp_path / "global.mnc"
    path.write_bytes(payload)
    meta = iio.immeta(path)
    assert meta["global:ident"] == "MINC test file"


# ---------------------------------------------------------------------------
# Compressed variants
# ---------------------------------------------------------------------------


def test_gzip_compressed(tmp_path, simple_volume):
    raw_path, data = simple_volume
    compressed = tmp_path / "volume.mnc.gz"
    compressed.write_bytes(gzip.compress(raw_path.read_bytes()))

    out = iio.imread(compressed)
    np.testing.assert_allclose(out, data.astype(np.float64) * 0.5 + 1.0)

    # also works when the extension does not advertise compression
    renamed = tmp_path / "volume_gz.mnc"
    renamed.write_bytes(compressed.read_bytes())
    np.testing.assert_allclose(iio.imread(renamed), out)


def test_bzip2_compressed(tmp_path, simple_volume):
    raw_path, data = simple_volume
    compressed = tmp_path / "volume.mnc.bz2"
    compressed.write_bytes(bz2.compress(raw_path.read_bytes()))

    out = iio.imread(compressed)
    np.testing.assert_allclose(out, data.astype(np.float64) * 0.5 + 1.0)


def test_read_from_bytes(simple_volume):
    _, data = simple_volume
    blob = simple_volume[0].read_bytes()
    out = iio.imread(blob, extension=".mnc")
    np.testing.assert_allclose(out, data.astype(np.float64) * 0.5 + 1.0)


# ---------------------------------------------------------------------------
# Distinguishable error conditions
# ---------------------------------------------------------------------------


def test_unknown_voxel_type(tmp_path):
    writer = NetCDFWriter()
    writer.add_dimension("yspace", 2).add_dimension("xspace", 2)
    # NetCDF type code 7 does not exist
    writer.add_variable("image", ["yspace", "xspace"], (), 7, np.ones((2, 2)))
    path = tmp_path / "badtype.mnc"
    path.write_bytes(writer.getvalue())

    assert isinstance(_open_explicit(path), MincTypeError)


def test_dimension_mismatch(tmp_path):
    # patch the image variable's dimension id to point past the last
    # dimension, so the variable references a non-existent dimension
    payload = bytearray(_make_minc(np.arange(2, dtype=">i2"), dims=("xspace",)))
    dimid_field = payload.index(b"image") + len(b"image") + 3 + 4
    assert payload[dimid_field : dimid_field + 4] == b"\x00\x00\x00\x00"
    payload[dimid_field : dimid_field + 4] = struct.pack(">I", 5)
    path = tmp_path / "mismatch.mnc"
    path.write_bytes(payload)

    assert isinstance(_open_explicit(path), MincFormatError)


def test_truncated_header(tmp_path):
    path = tmp_path / "trunc_header.mnc"
    path.write_bytes(b"CDF\x01\x00\x00\x00")  # cuts off inside the header
    assert isinstance(_open_explicit(path), MincTruncatedError)


def test_truncated_data(simple_volume):
    path, _ = simple_volume
    raw = path.read_bytes()
    truncated = path.with_name("trunc_data.mnc")
    truncated.write_bytes(raw[:-8])  # remove part of the voxel payload

    # header parses fine, the error surfaces when the data is read
    with iio.imopen(truncated, "r", plugin="MINC1") as img:
        with pytest.raises(MincTruncatedError):
            img.read()


def test_truncated_gzip(tmp_path, simple_volume):
    raw_path, _ = simple_volume
    blob = gzip.compress(raw_path.read_bytes())
    path = tmp_path / "trunc.mnc.gz"
    path.write_bytes(blob[: len(blob) // 2])

    # the cut may surface during header parsing or while the data is read,
    # depending on how much of the payload gzip can still decompress; either
    # way it is reported as a truncation
    try:
        img = iio.imopen(path, "r", plugin="MINC1")
    except OSError as exc:
        assert isinstance(exc.__cause__, MincTruncatedError)
    else:
        with pytest.raises(MincTruncatedError):
            img.read()


def test_corrupt_gzip_is_a_format_error(tmp_path):
    # a valid gzip header followed by garbage is corrupt, not truncated
    path = tmp_path / "corrupt.mnc.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 6 + b"\xde\xad\xbe\xef" * 4)
    assert isinstance(_open_explicit(path), MincFormatError)


def test_error_hierarchy():
    assert issubclass(MincFormatError, MincError)
    assert issubclass(MincTypeError, MincError)
    assert issubclass(MincTruncatedError, MincError)
    # all errors remain InitializationErrors so plugin fallback can proceed
    assert issubclass(MincError, InitializationError)
