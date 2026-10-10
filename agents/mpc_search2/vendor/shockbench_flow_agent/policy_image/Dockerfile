# The scoring server's policy container: the image a submission's agent.py runs in, one container per episode.
# Python 3.13, NumPy, SciPy, CPU torch, fastjsonschema and the agent kit, a non-root user (65534); at run time no
# network, a read-only file system, one CPU and 4 GB (shockbench_flow_agent.LIMITS.container). The kit is the source of
# the shockbench_flow_agent package (the shim) and of shockbench_flow (the protocol's codec and the flat view, with the
# public instances). SciPy and torch (the CPU build, no CUDA) are for participant code only: the kit imports neither.
# The container holds no scenario and no reference value.
# Built from a tar context of only this file, requirements.txt, launch.py (shockbench_flow/hosting/launch.py, standard
# library only) and lib/ (the kit's two packages): shockbench_flow_agent.build_context() makes it and
# shockbench_flow_agent.build_image() builds it. The base is the uv image referenced by digest (an OCI index;
# `docker build --platform linux/amd64` takes its amd64 manifest); the torch pin is the linux/amd64 wheel, so the image
# is linux/amd64 (emulated on an ARM machine).
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim@sha256:531f855bda2c73cd6ef67d56b733b357cea384185b3022bd09f05e002cd144ca

# the kit's shim module, run as `python -m <module> /submission` by launch.py (SBF_SHIM_MODULE)
ARG SHIM_MODULE=shockbench_flow_agent
# courtesy thread caps (set in the image as a courtesy, not as enforcement: --cpus 1 enforces): NumPy's BLAS and
# torch's intra-op pool read them; torch's inter-op pool reads none, so the kit's container entry pins it
# (shim.cap_torch_threads). Set before the checks below, which assert torch sees one thread.
ENV UV_PYTHON_DOWNLOADS=never UV_NO_CACHE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1
COPY requirements.txt /opt/sbf/requirements.txt
# every wheel by its hash (requirements.txt), no dependency resolution; bytecode compiled here, since the container runs
# `python -B` on a read-only root (without it each start would recompile torch's modules in memory)
RUN uv pip install --system --no-deps --require-hashes --compile-bytecode -r /opt/sbf/requirements.txt \
    && python -I -c "import numpy, scipy, fastjsonschema, torch; \
assert numpy.__version__ == '2.4.5', numpy.__version__; \
assert scipy.__version__ == '1.18.1', scipy.__version__; \
assert torch.__version__ == '2.14.0+cpu', torch.__version__; \
assert torch.version.cuda is None and not torch.cuda.is_available(), 'a CUDA torch in the policy image'; \
assert torch.get_num_threads() == 1, torch.get_num_threads()"
COPY launch.py /opt/sbf/launch.py
COPY lib/ /opt/sbf/lib/
# the kit compiled to bytecode (the containers run `python -B`, which reads bytecode but writes none), then imported
# and warmed here as the launcher will (a kit that needed a package the image lacks fails the build): its warm-up,
# when the module has one, is the first-use work the shim does before the ready line (shim.warm_up); neither may
# import torch or SciPy, which are participant code's
RUN test -n "$SHIM_MODULE" && python -I -m compileall -q /opt/sbf/lib && chmod -R a+rX,go-w /opt/sbf \
    && python -I -B -c "import importlib, importlib.util, sys; sys.path.insert(0, '/opt/sbf/lib'); \
spec = importlib.util.find_spec('$SHIM_MODULE'); assert spec is not None, 'shim $SHIM_MODULE not in lib/'; \
kit = importlib.import_module('$SHIM_MODULE') if spec.submodule_search_locations is not None else None; \
getattr(kit, 'warm_up', lambda: None)(); \
assert 'torch' not in sys.modules, 'the kit imports torch'; \
assert 'scipy' not in sys.modules, 'the kit imports SciPy'"

ENV SBF_SHIM_MODULE=${SHIM_MODULE} \
    SBF_KIT_PATH=/opt/sbf/lib \
    HOME=/tmp
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "-u", "/opt/sbf/launch.py"]
CMD ["/submission"]
