from __future__ import annotations

import unittest
from pathlib import Path

from app.hybrid_dashboard import _dashboard_href


class DashboardPathTests(unittest.TestCase):
    def test_dashboard_links_use_relative_paths_on_the_same_drive(self) -> None:
        href = _dashboard_href(
            Path(r"D:\project\diagnostics\fits.pdf"),
            Path(r"D:\project\results\hybrid"),
        )

        self.assertEqual(href, "../../diagnostics/fits.pdf")

    def test_dashboard_links_use_file_uri_across_drives(self) -> None:
        href = _dashboard_href(
            Path(r"D:\project\diagnostics\fits.pdf"),
            Path(r"C:\temporary\dashboard"),
        )

        self.assertEqual(href, "file:///D:/project/diagnostics/fits.pdf")


if __name__ == "__main__":
    unittest.main()
