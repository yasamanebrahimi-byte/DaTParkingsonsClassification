import zipfile

from datscan.data.discover import safe_extract_zip, discover_niftis


def test_safe_archive_extraction(tmp_path):
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("nested/a.nii.gz", b"test")
    destination = tmp_path / "out"
    safe_extract_zip(archive, destination)
    assert discover_niftis(destination)["a"]

