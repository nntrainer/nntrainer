#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host unit tests for the residue rules of native_stage.py.

    python3 Applications/CausalLM/Android/test_native_stage.py

The rules are exercised on file records, so no Android toolchain is needed.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_stage as ns  # noqa: E402

SYSTEM = ["libc.so", "libm.so", "libdl.so", "liblog.so"]


def lib(needed=None, kind="shared_library", **extra):
    record = {"kind": kind, "sha256": "0" * 64, "size": 1,
              "needed": list(needed if needed is not None else SYSTEM)}
    record.update(extra)
    return record


def core(extra=()):
    files = {name: lib() for name in ns.CORE_LIBRARIES}
    files["libc++_shared.so"] = lib()
    for name in extra:
        files[name] = lib()
    return files


def qnn_files():
    files = core(ns.QNN_REQUIRED)
    files["libQnnHtpV75Stub.so"] = lib()
    files["libQnnHtpV75Skel.so"] = {"kind": "dsp_image", "sha256": "0" * 64,
                                    "size": 1, "dsp_arch": "v75"}
    return files


def htp_files():
    files = core(["libsdkl.so"])
    files["libnntr_hvx_skel.so"] = {"kind": "dsp_image", "sha256": "0" * 64,
                                    "size": 1, "dsp_arch": "v81"}
    return files


def combined_files():
    files = qnn_files()
    files.update(htp_files())
    files["libQnnHtpV81Stub.so"] = lib()
    files["libQnnHtpV81Skel.so"] = {"kind": "dsp_image", "sha256": "0" * 64,
                                    "size": 1, "dsp_arch": "v81"}
    return files


def run(variant, kind, files):
    return ns.check_stage(variant, kind, files, Path("/nonexistent"), "readelf")


class ResidueRules(unittest.TestCase):
    def test_clean_stages_pass(self):
        self.assertEqual(run("cpu", "public", core()), [])
        self.assertEqual(run("qnn", "public", qnn_files()), [])
        self.assertEqual(run("htp", "public", htp_files()), [])
        self.assertEqual(run("qnn-htp", "public", combined_files()), [])

    def test_combined_requires_both_complete_runtimes_and_v81_pair(self):
        for missing in (*ns.QNN_REQUIRED, *ns.HTP_REQUIRED,
                        "libQnnHtpV81Stub.so", "libQnnHtpV81Skel.so"):
            files = combined_files()
            del files[missing]
            self.assertTrue(any(missing in p or "incomplete" in p
                                for p in run("qnn-htp", "public", files)), missing)

    def test_combined_checks_both_actual_dsp_architectures(self):
        for name in ("libQnnHtpV81Skel.so", "libnntr_hvx_skel.so"):
            files = combined_files()
            files[name]["dsp_arch"] = "v79"
            self.assertTrue(any("v81" in p for p in run("qnn-htp", "public", files)))

    def test_combined_rejects_an_unresolved_runtime_dependency(self):
        files = combined_files()
        files["libsdkl.so"]["needed"].append("libmissing.so")
        self.assertTrue(any("libmissing.so" in p for p in run("qnn-htp", "public", files)))

    def test_cpu_rejects_qnn_and_htp_residue(self):
        for leftover in ("libqnn_context.so", "libQnnHtp.so", "libsdkl.so",
                         "libnntr_hvx_skel.so"):
            problems = run("cpu", "public", core([leftover]))
            self.assertTrue(any(leftover in p for p in problems), leftover)

    def test_qnn_needs_its_runtime_and_a_stub_skel_pair(self):
        files = qnn_files()
        del files["libqnn_context.so"]
        self.assertTrue(any("libqnn_context.so" in p for p in run("qnn", "public", files)))
        files = qnn_files()
        del files["libQnnHtpV75Skel.so"]
        problems = run("qnn", "public", files)
        self.assertTrue(any("incomplete" in p for p in problems))
        self.assertTrue(any("no complete" in p for p in problems))

    def test_qnn_rejects_htp_libraries(self):
        files = qnn_files()
        files["libsdkl.so"] = lib()
        self.assertTrue(any("libsdkl.so" in p for p in run("qnn", "public", files)))

    def test_htp_needs_sdkl_and_a_v81_skel_and_rejects_qnn(self):
        files = htp_files()
        del files["libsdkl.so"]
        self.assertTrue(any("libsdkl.so" in p for p in run("htp", "public", files)))
        files = htp_files()
        files["libnntr_hvx_skel.so"]["dsp_arch"] = "v75"
        self.assertTrue(any("v81" in p for p in run("htp", "public", files)))
        files = htp_files()
        files["libqnn_context.so"] = lib()
        self.assertTrue(any("libqnn_context.so" in p for p in run("htp", "public", files)))

    def test_fastrpc_stub_never_ships(self):
        for variant, files in (("cpu", core()), ("qnn", qnn_files()), ("htp", htp_files()),
                               ("qnn-htp", combined_files())):
            files = dict(files)
            files["libcdsprpc.so"] = lib()
            self.assertTrue(any("libcdsprpc.so" in p for p in run(variant, "public", files)))

    def test_exactly_one_libcxx(self):
        files = core()
        del files["libc++_shared.so"]
        self.assertTrue(any("libc++_shared.so" in p for p in run("cpu", "public", files)))

    def test_public_stage_has_no_plugin_internal_needs_it(self):
        self.assertTrue(any(ns.PLUGIN in p
                            for p in run("cpu", "public", core([ns.PLUGIN]))))
        self.assertTrue(any(ns.PLUGIN in p for p in run("cpu", "internal", core())))

    def test_retired_legacy_library_is_residue(self):
        self.assertTrue(any("libquick_dot_ai.so" in p
                            for p in run("cpu", "public", core(["libquick_dot_ai.so"]))))

    def test_unresolved_dependency(self):
        files = core()
        files["libcausallm.so"] = lib(SYSTEM + ["libmissing.so"])
        self.assertTrue(any("libmissing.so" in p for p in run("cpu", "public", files)))


class DescribeStage(unittest.TestCase):
    def test_marker_and_gitkeep_are_not_stage_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ns.NOT_SHIPPED:
                (Path(tmp) / name).write_text("")
            self.assertEqual(ns.describe_stage(Path(tmp), "unused-readelf"), {})


class ManifestDrift(unittest.TestCase):
    def test_stale_and_changed_files_are_reported(self):
        files = {"a.so": {"sha256": "1"}, "stale.so": {"sha256": "2"}}
        manifest = {"libraries": {"a.so": {"sha256": "9"}, "gone.so": {"sha256": "3"}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            path.write_text(json.dumps(manifest))
            problems = ns.compare_with_manifest(files, path)
        self.assertTrue(any("stale stale.so" in p for p in problems))
        self.assertTrue(any("missing gone.so" in p for p in problems))
        self.assertTrue(any("a.so: sha256" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
