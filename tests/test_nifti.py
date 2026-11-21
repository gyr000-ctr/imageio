"""Tests for the NIfTI-1 plugin."""

import gzip
from io import BytesIO

import numpy as np
import pytest

import imageio.v3 as iio
from imageio.config import known_extensions, known_plugins
from imageio.plugins.nifti import (
    HEADER_DTYPE,
    HEADER_SIZE,
    MIN_VOX_OFFSET,
    NiftiFormatError,
    NiftiPlugin,
    TruncatedNiftiError,
    UnknownNiftiDatatypeError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_header(
    shape=(4, 5, 6),
    datatype=16,
    bitpix=32,
    pixdim=(1.0, 1.0, 1.0, 1.0),
    sform=None,
    qform=None,
    scl_slope=1.0,
    scl_inter=0.0,
    magic=b"n+1\x00",
    vox_offset=MIN_VOX_OFFSET,
    byteorder="<",
):
    """Create a raw 348-byte NIfTI-1 header for testing."""
    header = np.zeros(1, dtype=HEADER_DTYPE.newbyteorder(byteorder))
    header["sizeof_hdr"] = HEADER_SIZE
    header["magic"] = magic
    header["regular"] = b"r"

    ndim = len(shape)
    dim = [ndim] + list(shape) + [1] * (7 - ndim)
    header["dim"] = np.asarray(dim, dtype=np.int16)

    header["datatype"] = datatype
    header["bitpix"] = bitpix

    # ``pixdim`` is given in NIfTI's own layout, i.e. pixdim[0] is the qfac
    # followed by the three spatial spacings.
    full_pixdim = list(pixdim) + [1.0] * (8 - len(pixdim))
    header["pixdim"] = np.asarray(full_pixdim, dtype="<f4")

    header["vox_offset"] = float(vox_offset)
    header["scl_slope"] = float(scl_slope)
    header["scl_inter"] = float(scl_inter)

    if sform is not None:
        affine = np.asarray(sform, dtype=np.float64)
        header["sform_code"] = 1
        header["srow_x"] = np.asarray(affine[0, :4], dtype="<f4")
        header["srow_y"] = np.asarray(affine[1, :4], dtype="<f4")
        header["srow_z"] = np.asarray(affine[2, :4], dtype="<f4")
    else:
        header["sform_code"] = 0

    if qform is not None:
        (
            b,
            c,
            d,
            ox,
            oy,
            oz,
            qfac,
        ) = qform
        header["qform_code"] = 1
        header["quatern_b"] = b
        header["quatern_c"] = c
        header["quatern_d"] = d
        header["qoffset_x"] = ox
        header["qoffset_y"] = oy
        header["qoffset_z"] = oz
        # ``header["pixdim"]`` has shape (1, 8); index both axes so this only
        # changes pixdim[0].
        header["pixdim"][0, 0] = qfac
    else:
        header["qform_code"] = 0

    return header.tobytes()


def make_nifti(
    data,
    *,
    gzipped=False,
    **header_kwargs,
):
    data = np.asarray(data)
    if "datatype" not in header_kwargs:
        dtype_to_code = {
            np.dtype("u1"): 2,
            np.dtype("i2"): 4,
            np.dtype("i4"): 8,
            np.dtype("f4"): 16,
            np.dtype("f8"): 64,
            np.dtype("i1"): 256,
            np.dtype("u2"): 512,
            np.dtype("u4"): 768,
            np.dtype("i8"): 1024,
            np.dtype("u8"): 1280,
        }
        header_kwargs["datatype"] = dtype_to_code[data.dtype]
        header_kwargs.setdefault("bitpix", data.dtype.itemsize * 8)
    header_kwargs.setdefault("shape", data.shape)
    byteorder = header_kwargs.get("byteorder", "<")

    raw = (
        make_header(**header_kwargs)
        + b"\x00\x00\x00\x00"
        # NIfTI stores the first axis as the fastest-varying one.
        + data.astype(data.dtype.newbyteorder(byteorder)).tobytes(order="F")
    )
    if gzipped:
        return gzip.compress(raw)
    return raw


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def test_plugin_is_registered():
    assert "NIFTI" in known_plugins
    assert known_plugins["NIFTI"].class_name == "NiftiPlugin"

    # The existing plugins keep their priority; NIFTI is preferred for nii.
    for extension in (".nii", ".nii.gz"):
        priorities = [
            plugin
            for file_ext in known_extensions[extension]
            for plugin in file_ext.priority
        ]
        assert "NIFTI" in priorities
        assert "ITK" in priorities
        assert priorities.index("NIFTI") < priorities.index("ITK")


def test_existing_plugin_order_is_preserved():
    from imageio.config.plugins import _original_order

    # The legacy plugins are the original ones and must be untouched.
    assert _original_order[:5] == [
        "TIFF",
        "BMP-PIL",
        "BUFR-PIL",
        "CUR-PIL",
        "DCX-PIL",
    ]
    assert "NIFTI" not in _original_order

    # Other existing extensions still resolve to their original plugins.
    png_priorities = [
        plugin for ext in known_extensions[".png"] for plugin in ext.priority
    ]
    assert png_priorities[0] == "pillow"


def test_gz_extension_is_routed_to_nifti():
    """`.nii.gz` resolves to extension `.gz`, which must reach NIFTI."""
    priorities = [plugin for ext in known_extensions[".gz"] for plugin in ext.priority]
    assert priorities[0] == "NIFTI"


def test_plugin_name_and_extension():
    with iio.imopen(
        BytesIO(make_nifti(np.zeros((2, 2), np.uint8))), "r", plugin="NIFTI"
    ) as f:
        assert isinstance(f, NiftiPlugin)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def test_read_plain(tmp_path):
    data = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    path = tmp_path / "test.nii"
    path.write_bytes(make_nifti(data))

    result = iio.imread(path)
    np.testing.assert_array_equal(result, data)
    # The default scaling is 1.0/0.0, so the native dtype is preserved.
    assert result.dtype == np.int16


def test_read_gzip(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "test.nii.gz"
    path.write_bytes(make_nifti(data, gzipped=True))

    result = iio.imread(path)
    np.testing.assert_array_equal(result, data)
    assert result.dtype == np.float32


def test_read_gzip_compressed_by_content(tmp_path):
    """A gzip stream in a `.nii` file is still read correctly."""
    data = np.ones((3, 3, 3), np.uint16)
    path = tmp_path / "test.nii"
    path.write_bytes(make_nifti(data, gzipped=True))

    np.testing.assert_array_equal(iio.imread(path), data)


def test_data_is_stored_first_axis_fastest(tmp_path):
    """The array axes line up with the affine (nibabel's (i, j, k) order).

    Writing an array whose value encodes its index and re-reading it must
    reproduce the same array, no axis permutation.
    """
    shape = (2, 3, 4)
    data = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    path = tmp_path / "order.nii"
    iio.imwrite(path, data)

    # In the raw file the first axis varies fastest.
    raw = path.read_bytes()[MIN_VOX_OFFSET:]
    stored = np.frombuffer(raw, dtype="<i2")[: data.size]
    np.testing.assert_array_equal(stored, data.ravel(order="F"))

    # Reading gives back the array unchanged.
    np.testing.assert_array_equal(iio.imread(path), data)


def test_read_from_bytes():
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    result = iio.imread(make_nifti(data))
    np.testing.assert_array_equal(result, data)


def test_read_all_supported_dtypes(tmp_path):
    code_to_dtype = {
        2: np.uint8,
        4: np.int16,
        8: np.int32,
        16: np.float32,
        64: np.float64,
        256: np.int8,
        512: np.uint16,
        768: np.uint32,
        1024: np.int64,
        1280: np.uint64,
    }
    for code, dtype in code_to_dtype.items():
        data = np.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
        path = tmp_path / f"test_{code}.nii"
        path.write_bytes(
            make_nifti(data, datatype=code, bitpix=np.dtype(dtype).itemsize * 8)
        )

        result = iio.imread(path)
        np.testing.assert_array_equal(result, data)
        assert result.dtype == np.dtype(dtype)


def test_read_scaling_applied_by_default(tmp_path):
    data = np.arange(12, dtype=np.int16).reshape(2, 2, 3)
    path = tmp_path / "scaled.nii"
    path.write_bytes(make_nifti(data, scl_slope=2.5, scl_inter=-10.0))

    result = iio.imread(path)
    np.testing.assert_allclose(result, data * 2.5 - 10.0)
    assert result.dtype == np.float64


def test_read_scaling_opt_out(tmp_path):
    data = np.arange(12, dtype=np.int16).reshape(2, 2, 3)
    path = tmp_path / "scaled.nii.gz"
    path.write_bytes(make_nifti(data, gzipped=True, scl_slope=2.5, scl_inter=-10.0))

    result = iio.imread(path, apply_scaling=False)
    np.testing.assert_array_equal(result, data)
    assert result.dtype == np.int16


def test_read_zero_slope_means_no_scaling(tmp_path):
    data = np.arange(6, dtype=np.int16).reshape(1, 2, 3)
    path = tmp_path / "zero_slope.nii"
    path.write_bytes(make_nifti(data, scl_slope=0.0, scl_inter=5.0))

    result = iio.imread(path)
    np.testing.assert_array_equal(result, data)
    assert result.dtype == np.int16


def test_read_index_semantics(tmp_path):
    data = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    path = tmp_path / "single.nii"
    path.write_bytes(make_nifti(data))

    np.testing.assert_array_equal(iio.imread(path, index=0), data)
    np.testing.assert_array_equal(iio.imread(path, index=...), data)

    with pytest.raises(IndexError):
        iio.imread(path, index=1)


def test_iter_and_properties(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    path = tmp_path / "props.nii"
    path.write_bytes(make_nifti(data, pixdim=(1.0, 0.5, 0.25, 2.0)))

    images = list(iio.imiter(path))
    assert len(images) == 1
    np.testing.assert_array_equal(images[0], data)

    props = iio.improps(path)
    assert props.shape == (2, 3, 4)
    assert props.dtype == np.float32
    assert props.is_batch is False
    np.testing.assert_allclose(props.spacing, (0.5, 0.25, 2.0))


def test_properties_reports_scaled_dtype(tmp_path):
    data = np.zeros((2, 2, 2), np.int16)
    path = tmp_path / "scaled_props.nii"
    path.write_bytes(make_nifti(data, scl_slope=2.0))

    props = iio.improps(path)
    assert props.dtype == np.float64


def test_metadata_reports_spatial_info(tmp_path):
    data = np.zeros((2, 3, 4), np.float32)
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -10.0],
            [0.0, 3.0, 0.0, 5.0],
            [0.0, 0.0, 4.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = tmp_path / "meta.nii"
    path.write_bytes(make_nifti(data, sform=affine, pixdim=(1.0, 2.0, 3.0, 4.0)))

    meta = iio.immeta(path)
    np.testing.assert_allclose(meta["affine"], affine)
    np.testing.assert_allclose(meta["sform"], affine)
    assert meta["dim"] == [3, 2, 3, 4, 1, 1, 1, 1]
    assert meta["datatype"] == 16
    assert meta["bitpix"] == 32

    # Applied metadata is hidden by default ...
    assert "scl_slope" not in meta
    # ... unless explicitly requested.
    meta_all = iio.immeta(path, exclude_applied=False)
    assert meta_all["scl_slope"] == 1.0
    assert meta_all["scl_inter"] == 0.0


def test_qform_only_file(tmp_path):
    # A qform describing an identity rotation with 2 mm voxels and an offset.
    qform = (0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0)
    data = np.zeros((2, 2, 2), np.int16)
    path = tmp_path / "qform.nii"
    path.write_bytes(make_nifti(data, qform=qform, pixdim=(1.0, 2.0, 2.0, 2.0)))

    meta = iio.immeta(path)
    assert "qform" in meta
    assert "sform" not in meta

    expected = np.array(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(meta["affine"], expected)
    np.testing.assert_allclose(meta["qform"], expected)


def test_prefer_sform_over_qform(tmp_path):
    sform = np.array(
        [
            [1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 200.0],
            [0.0, 0.0, 1.0, 300.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    qform = (0.0, 0.0, 0.0, -1.0, -2.0, -3.0, 1.0)
    data = np.zeros((2, 2, 2), np.int16)
    path = tmp_path / "both.nii"
    path.write_bytes(
        make_nifti(data, sform=sform, qform=qform, pixdim=(1.0, 1.0, 1.0, 1.0))
    )

    meta = iio.immeta(path)
    np.testing.assert_allclose(meta["affine"], sform)
    # Both are still exposed individually.
    assert "sform" in meta and "qform" in meta


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_roundtrip_nii(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, -5.0],
            [0.0, 0.0, 3.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = tmp_path / "out.nii"
    iio.imwrite(path, data, affine=affine, scl_slope=1.0, scl_inter=0.0)

    result = iio.imread(path)
    np.testing.assert_array_equal(result, data)

    meta = iio.immeta(path)
    np.testing.assert_allclose(meta["affine"], affine, atol=1e-6)
    np.testing.assert_allclose(meta["sform"], affine, atol=1e-6)


def test_roundtrip_niigz(tmp_path):
    data = np.random.default_rng(0).random((3, 4, 5)).astype(np.float32)
    path = tmp_path / "out.nii.gz"
    iio.imwrite(path, data)

    assert path.read_bytes()[:2] == b"\x1f\x8b"
    np.testing.assert_array_equal(iio.imread(path), data)


def test_write_preserves_slope_and_intercept(tmp_path):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "scaled_out.nii"
    iio.imwrite(path, data, scl_slope=4.0, scl_inter=100.0)

    raw = iio.imread(path, apply_scaling=False)
    np.testing.assert_array_equal(raw, data)

    scaled = iio.imread(path)
    np.testing.assert_allclose(scaled, data * 4.0 + 100.0)

    meta = iio.immeta(path, exclude_applied=False)
    assert meta["scl_slope"] == pytest.approx(4.0)
    assert meta["scl_inter"] == pytest.approx(100.0)


def test_write_with_affine_and_scaling_roundtrip(tmp_path):
    """The affine and slope/intercept survive a read-modify-write cycle."""
    data = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    affine = np.array(
        [
            [0.0, -2.0, 0.0, 1.5],
            [3.0, 0.0, 0.0, -2.5],
            [0.0, 0.0, 4.0, 9.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    first = tmp_path / "first.nii"
    iio.imwrite(first, data, affine=affine, scl_slope=2.0, scl_inter=1.0)

    # Read back the spatial info and scaling, then write it again.
    meta = iio.immeta(first, exclude_applied=False)
    second = tmp_path / "second.nii"
    iio.imwrite(
        second,
        iio.imread(first, apply_scaling=False),
        affine=meta["affine"],
        scl_slope=meta["scl_slope"],
        scl_inter=meta["scl_inter"],
    )

    meta2 = iio.immeta(second, exclude_applied=False)
    np.testing.assert_allclose(meta2["affine"], affine, atol=1e-5)
    assert meta2["scl_slope"] == pytest.approx(meta["scl_slope"])
    assert meta2["scl_inter"] == pytest.approx(meta["scl_inter"])
    np.testing.assert_array_equal(iio.imread(first), iio.imread(second))


def test_write_affine_3x4(tmp_path):
    data = np.zeros((2, 2, 2), np.uint8)
    affine = np.array(
        [[1.0, 0.0, 0.0, 4.0], [0.0, 1.0, 0.0, 5.0], [0.0, 0.0, 1.0, 6.0]]
    )
    path = tmp_path / "affine3x4.nii"
    iio.imwrite(path, data, affine=affine)

    meta = iio.immeta(path)
    np.testing.assert_allclose(meta["affine"], np.vstack([affine, [0, 0, 0, 1]]))


def test_write_pixdim_and_metadata(tmp_path):
    data = np.zeros((2, 2, 2), np.float32)
    path = tmp_path / "pixdim.nii"
    iio.imwrite(
        path,
        data,
        pixdim=(0.5, 0.75, 1.25),
        metadata={"descrip": "hello nifti", "xyzt_units": 10},
    )

    meta = iio.immeta(path)
    np.testing.assert_allclose(meta["pixdim"][1:4], (0.5, 0.75, 1.25))
    assert meta["descrip"] == "hello nifti"
    assert meta["xyzt_units"] == 10


def test_write_to_bytes_target():
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    encoded = iio.imwrite("<bytes>", data, plugin="NIFTI")

    assert isinstance(encoded, bytes)
    np.testing.assert_array_equal(iio.imread(encoded), data)


def test_write_rejects_unknown_metadata_key(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        iio.imwrite(
            tmp_path / "out.nii", np.zeros((2, 2), np.float32), metadata={"nope": 1}
        )


def test_write_rejects_unsupported_dtype(tmp_path):
    with pytest.raises(ValueError, match="cannot store"):
        iio.imwrite(tmp_path / "out.nii", np.zeros((2, 2), np.complex64))


def test_write_rejects_bad_ndim(tmp_path):
    with pytest.raises(ValueError, match="dimensions"):
        iio.imwrite(tmp_path / "out.nii", np.zeros((2,) * 8, np.float32))


def test_write_bool_is_stored_as_uint8(tmp_path):
    data = np.array([[True, False], [False, True]])
    path = tmp_path / "bool.nii"
    iio.imwrite(path, data)
    np.testing.assert_array_equal(iio.imread(path), data.astype(np.uint8))


def test_write_other_extension_raises(tmp_path):
    with pytest.raises(OSError):
        iio.imopen(tmp_path / "out.nrrd", "w", plugin="NIFTI")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_bad_magic(tmp_path):
    data = np.zeros((2, 2), np.int16)
    path = tmp_path / "bad_magic.nii"
    path.write_bytes(make_nifti(data, magic=b"n+2\x00"))

    with pytest.raises(NiftiFormatError, match="magic"):
        iio.imread(path)

    # It is both an OSError and a ValueError, and is *not* a truncation or
    # datatype error, so the errors stay distinguishable.
    with pytest.raises(OSError):
        iio.imread(path)
    with pytest.raises(ValueError):
        iio.imread(path)
    with pytest.raises(NiftiFormatError) as info:
        iio.imread(path)
    assert not isinstance(info.value, TruncatedNiftiError)
    assert not isinstance(info.value, UnknownNiftiDatatypeError)


def test_pair_file_is_rejected(tmp_path):
    data = np.zeros((2, 2), np.int16)
    path = tmp_path / "pair.nii"
    path.write_bytes(make_nifti(data, magic=b"ni1\x00"))

    with pytest.raises(NiftiFormatError, match="ni1"):
        iio.imread(path)


def test_header_too_short(tmp_path):
    path = tmp_path / "short.nii"
    path.write_bytes(b"n+1\x00" * 10)

    # An incomplete header is a truncation, not a magic error.
    with pytest.raises(TruncatedNiftiError, match="header"):
        iio.imread(path)


def test_big_endian_file(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "big_endian.nii"
    path.write_bytes(make_nifti(data, byteorder=">"))

    result = iio.imread(path)
    np.testing.assert_array_equal(result, data)
    assert result.dtype.byteorder in ("=", "|", "<") or result.dtype.itemsize == 1


def test_big_endian_scaling_and_spacing(tmp_path):
    data = np.arange(2 * 2 * 3, dtype=np.float32).reshape(2, 2, 3)
    path = tmp_path / "big_endian_scaled.nii"
    path.write_bytes(
        make_nifti(
            data,
            byteorder=">",
            scl_slope=2.0,
            scl_inter=1.0,
            pixdim=(1.0, 0.5, 0.25, 2.0),
        )
    )

    np.testing.assert_allclose(iio.imread(path), data * 2.0 + 1.0)
    np.testing.assert_allclose(iio.improps(path).spacing, (0.5, 0.25, 2.0))


def test_nifti2_is_rejected(tmp_path):
    """A NIfTI-2 header (sizeof_hdr == 540) gets a specific message."""
    path = tmp_path / "nifti2.nii"
    path.write_bytes((540).to_bytes(4, "little") + bytes(600))

    with pytest.raises(NiftiFormatError, match="NIfTI-2"):
        iio.imread(path)


def test_truncated_data(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    raw = make_nifti(data)
    path = tmp_path / "truncated.nii"
    # Keep the complete header but cut the voxel data in half.
    path.write_bytes(raw[: MIN_VOX_OFFSET + raw[MIN_VOX_OFFSET:].__len__() // 2])

    with pytest.raises(TruncatedNiftiError) as info:
        iio.imread(path)
    assert info.value.expected == data.nbytes
    assert info.value.actual == data.nbytes // 2
    assert info.value.actual < info.value.expected


def test_truncated_gzip_data(tmp_path):
    data = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    raw = make_nifti(data)
    path = tmp_path / "truncated.nii.gz"
    path.write_bytes(gzip.compress(raw[: MIN_VOX_OFFSET + data.nbytes // 2]))

    with pytest.raises(TruncatedNiftiError) as info:
        iio.imread(path)
    assert info.value.actual == data.nbytes // 2


def test_bad_magic_in_gzip(tmp_path):
    raw = bytearray(make_nifti(np.zeros((2, 2), np.int16)))
    raw[344:348] = b"xxxx"
    path = tmp_path / "bad_magic.nii.gz"
    path.write_bytes(gzip.compress(bytes(raw)))

    with pytest.raises(NiftiFormatError, match="magic") as info:
        iio.imread(path)
    assert not isinstance(info.value, TruncatedNiftiError)


def test_non_nifti_gzip_is_not_claimed(tmp_path):
    """A `.gz` file that is not NIfTI must not be claimed by the plugin."""
    path = tmp_path / "data.gz"
    path.write_bytes(gzip.compress(b"This is not a NIfTI-1 file at all."))

    with pytest.raises(OSError):
        iio.imread(path)

    # Explicitly selecting the plugin fails at initialization, like it does
    # for other content-sniffing plugins (e.g. SPE).
    with pytest.raises(OSError):
        iio.imopen(path, "r", plugin="NIFTI")


def test_corrupt_file_with_nifti_name_raises_format_error(tmp_path):
    """A NIfTI-named file gets the distinguishable error, not a bare OSError."""
    path = tmp_path / "corrupt.nii.gz"
    path.write_bytes(gzip.compress(b"This is not a NIfTI-1 file at all."))

    with pytest.raises(NiftiFormatError):
        iio.imread(path)


def test_unknown_datatype(tmp_path):
    data = np.zeros((2, 2), np.uint8)
    path = tmp_path / "unknown_dtype.nii"
    # 9999 is not a valid NIfTI-1 datatype code.
    path.write_bytes(make_nifti(data, datatype=9999, bitpix=8))

    with pytest.raises(UnknownNiftiDatatypeError) as info:
        iio.imread(path)
    assert info.value.datatype == 9999
    assert not isinstance(info.value, TruncatedNiftiError)


def test_packed_rgb_datatype_is_reported_as_unknown(tmp_path):
    data = np.zeros((2, 2, 3), np.uint8)
    path = tmp_path / "rgb.nii"
    path.write_bytes(make_nifti(data, datatype=128, bitpix=24))

    with pytest.raises(UnknownNiftiDatatypeError) as info:
        iio.imread(path)
    assert info.value.datatype == 128


def test_errors_are_distinguishable(tmp_path):
    good = make_nifti(np.zeros((2, 2), np.int16))

    bad_magic = tmp_path / "a.nii"
    bad_magic.write_bytes(good[:344] + b"xxxx" + good[348:])

    truncated = tmp_path / "b.nii"
    truncated.write_bytes(good[:-4])

    unknown = tmp_path / "c.nii"
    unknown.write_bytes(make_nifti(np.zeros((2, 2), np.uint8), datatype=9999, bitpix=8))

    errors = []
    for path in (bad_magic, truncated, unknown):
        with pytest.raises(NiftiFormatError) as info:
            iio.imread(path)
        errors.append(type(info.value))

    assert errors == [NiftiFormatError, TruncatedNiftiError, UnknownNiftiDatatypeError]


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


def test_auto_discovery_for_nii_extensions(tmp_path):
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    for suffix in (".nii", ".nii.gz"):
        path = tmp_path / ("auto" + suffix)
        iio.imwrite(path, data)
        with iio.imopen(path, "r") as f:
            assert isinstance(f, NiftiPlugin)
        np.testing.assert_array_equal(iio.imread(path), data)


def test_explicit_plugin_override(tmp_path):
    data = np.zeros((2, 2), np.int16)
    path = tmp_path / "explicit.nii"
    path.write_bytes(make_nifti(data))

    with iio.imopen(path, "r", plugin="NIFTI") as f:
        np.testing.assert_array_equal(f.read(), data)


def test_non_nifti_file_falls_through(tmp_path):
    """A `.nii` file that is not NIfTI raises a distinguishable error."""
    path = tmp_path / "not_really.nii"
    path.write_bytes(b"This is definitely not a NIfTI file." + bytes(400))

    with pytest.raises(NiftiFormatError):
        iio.imread(path)


def test_legacy_api_read(tmp_path):
    import imageio.v2 as iio_v2

    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "legacy.nii"
    path.write_bytes(make_nifti(data))

    result = iio_v2.imread(path)
    np.testing.assert_array_equal(result, data)

    result = iio_v2.imread(path, format="NIFTI")
    np.testing.assert_array_equal(result, data)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_metadata_descrip_padded(tmp_path):
    data = np.zeros((2, 2), np.int16)
    path = tmp_path / "descrip.nii"
    path.write_bytes(make_nifti(data))

    # Nothing stored: metadata returns an empty string, not NUL bytes.
    meta = iio.immeta(path)
    assert meta["descrip"] == ""
    assert meta["aux_file"] == ""


@pytest.mark.parametrize("suffix", [".nii", ".nii.gz"])
def test_cross_validate_with_nibabel(tmp_path, suffix):
    """Files interoperate with nibabel in both directions.

    This independently confirms the on-disk byte order and the affine
    convention. Skipped when nibabel is not installed.
    """
    nib = pytest.importorskip("nibabel")

    data = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4) - 5
    affine = np.array(
        [
            [0.0, -1.5, 0.0, 12.0],
            [1.25, 0.0, 0.0, -30.0],
            [0.0, 0.0, 2.0, 44.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # nibabel -> NIFTI plugin
    nb_path = tmp_path / ("from_nibabel" + suffix)
    nib.save(nib.Nifti1Image(data, affine), str(nb_path))

    np.testing.assert_array_equal(iio.imread(nb_path, apply_scaling=False), data)
    np.testing.assert_allclose(iio.immeta(nb_path)["affine"], affine, atol=1e-5)

    # NIFTI plugin -> nibabel
    our_path = tmp_path / ("from_plugin" + suffix)
    iio.imwrite(our_path, data, affine=affine)

    reloaded = nib.load(str(our_path))
    np.testing.assert_array_equal(np.asanyarray(reloaded.dataobj), data)
    np.testing.assert_allclose(reloaded.affine, affine, atol=1e-5)
