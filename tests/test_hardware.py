from __future__ import annotations

import unittest
from unittest import mock

from taterbench.hardware import _memory_bytes, _strix_installed_memory_bytes, is_strix_halo


class HardwareTests(unittest.TestCase):
    def test_strix_identity_detection(self) -> None:
        with (
            mock.patch("taterbench.hardware.platform.system", return_value="Linux"),
            mock.patch(
                "taterbench.hardware._cpu_name",
                return_value="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S",
            ),
            mock.patch("taterbench.hardware.Path.read_text", side_effect=OSError),
        ):
            self.assertTrue(is_strix_halo())

    def test_strix_uma_reservation_reconstructs_installed_memory(self) -> None:
        visible = 15_470_456 * 1024
        reserved_vram = 48 * 1024**3
        self.assertEqual(_strix_installed_memory_bytes(visible, reserved_vram), 64 * 1024**3)

    def test_memory_uses_visible_plus_reserved_vram_only_on_strix(self) -> None:
        visible = 15_470_456 * 1024
        reserved_vram = 48 * 1024**3
        with (
            mock.patch("taterbench.hardware.platform.system", return_value="Linux"),
            mock.patch("taterbench.hardware._visible_memory_bytes", return_value=visible),
            mock.patch("taterbench.hardware._strix_reserved_vram_bytes", return_value=reserved_vram),
            mock.patch("taterbench.hardware.is_strix_halo", return_value=True),
        ):
            self.assertEqual(_memory_bytes(), 64 * 1024**3)
        with (
            mock.patch("taterbench.hardware.platform.system", return_value="Linux"),
            mock.patch("taterbench.hardware._visible_memory_bytes", return_value=visible),
            mock.patch("taterbench.hardware.is_strix_halo", return_value=False),
        ):
            self.assertEqual(_memory_bytes(), visible)


if __name__ == "__main__":
    unittest.main()
