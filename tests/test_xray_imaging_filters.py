from dpf2.diagnostics.xray_imaging import (
    apply_filter_pack,
    pinhole_camera,
)


def test_apply_filter_pack_be():
    energies = [1.0, 2.0]
    filtered = apply_filter_pack(energies, 'Be')
    assert filtered[0] == 0.5  # 1.0 * 0.5
    assert filtered[1] == 0.4  # 2.0 * 0.2


def test_pinhole_camera_with_filter():
    positions = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
    energies = [1.0, 1.0]
    image = pinhole_camera(
        positions,
        energies,
        detector_distance=1.0,
        detector_pixels=(1, 1),
        pixel_size=1.0,
        filter_pack='Be',
    )
    # Each photon is attenuated to 0.5 -> total 1.0 in single pixel
    assert image == [[1.0]]
