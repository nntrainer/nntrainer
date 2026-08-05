import os
from _util import load
nenv = load("nntrainer_env_test", "agents/nntrainer_env.py")  # stdlib only


def test_prefix_discovery(tmp_path=None):
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "include"))
        cflags, libs, source = nenv.discover_flags(prefix=d)
        assert cflags[:1] == ["-I"]
        assert "-lccapi-nntrainer" in libs and "-lnntrainer" in libs
        assert source.startswith("prefix")


def test_env_discovery_multi_include():
    os.environ["NNTRAINER_INCLUDE_DIR"] = os.pathsep.join(["/opt/a/include", "/opt/b/include"])
    os.environ["NNTRAINER_LIB_DIR"] = "/opt/a/lib"
    try:
        cflags, libs, source = nenv.discover_flags()
        assert cflags.count("-I") == 2
        assert "-L" in libs
        assert libs[-2:] == ["-lccapi-nntrainer", "-lnntrainer"]  # ccapi first
        assert source == "NNTRAINER_INCLUDE_DIR"
    finally:
        del os.environ["NNTRAINER_INCLUDE_DIR"]
        del os.environ["NNTRAINER_LIB_DIR"]


def test_not_found_returns_none():
    # no prefix, no env; pkg-config almost certainly has no nntrainer here
    for k in ("NNTRAINER_INCLUDE_DIR", "NNTRAINER_LIB_DIR"):
        os.environ.pop(k, None)
    assert nenv.discover_flags() == (None, None, None)


def test_multiarch_lib_dir_resolved():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "include"))
        arch = os.path.join(d, "lib", "aarch64-linux-gnu")
        os.makedirs(arch)
        open(os.path.join(arch, "libnntrainer.so"), "w").close()
        cflags, libs, _ = nenv.discover_flags(prefix=d)
        assert libs[libs.index("-L") + 1] == arch  # multiarch dir, not plain lib/
