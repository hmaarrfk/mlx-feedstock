# Verifies the separate NAX (Metal 4) shader library, mlx_nax.metallib, shipped
# by recipe/0002-*.patch. This whole mechanism -- the patch, this test, and the
# extra metallib -- can be removed once MACOSX_DEPLOYMENT_TARGET is raised past
# 26.2, because upstream then builds the NAX kernels straight into mlx.metallib.
import glob
import os
import sys

libdir = os.path.join(sys.prefix, "lib")
base = os.path.join(libdir, "mlx.metallib")
nax = os.path.join(libdir, "mlx_nax.metallib")

# The base Metal shader library must always ship next to libmlx.dylib.
found = glob.glob(os.path.join(sys.prefix, "**", "mlx.metallib"), recursive=True)
assert os.path.isfile(base), f"mlx.metallib not at {base}; found instead: {found}"

# The NAX metallib is built only when the build toolchain provides a macOS >=
# 26.2 Metal SDK (see the gate in recipe/0002-*.patch). When it is built it MUST
# be shipped right next to mlx.metallib -- the runtime loads it from there via
# get_library("mlx_nax") -- and be non-empty. Its absence on an older-SDK build
# is expected (the NAX kernels are compiled out and never dispatched).
if os.path.isfile(nax):
    assert os.path.getsize(nax) > 0, f"{nax} is empty"
    print(f"OK: mlx_nax.metallib shipped ({os.path.getsize(nax)} bytes) next to mlx.metallib")
else:
    stray = glob.glob(os.path.join(sys.prefix, "**", "mlx_nax.metallib"), recursive=True)
    assert not stray, f"mlx_nax.metallib built but not installed next to mlx.metallib: {stray}"
    print("OK: mlx_nax.metallib not built (build Metal SDK < 26.2) -- NAX compiled out, as expected")

# Exercise the attention path end to end. On a NAX-capable GPU + build this
# dispatches the NAX kernel and loads mlx_nax.metallib; were that library built
# but not packaged, this would raise "Failed to load the metallib mlx_nax.metallib".
import mlx.core as mx  # noqa: E402

q = mx.random.normal((1, 8, 256, 64))
out = mx.fast.scaled_dot_product_attention(q, q, q, scale=64 ** -0.5)
mx.eval(out)
assert out.shape == (1, 8, 256, 64)
print("OK: scaled_dot_product_attention ran on", mx.default_device())
