# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

First version, installed from a clone so far. It reads SPEC scans and Timepix TIFF frames through
QoraGeom, maps a rocking scan to q with the detector angles, finds streak and facet directions,
draws line profiles and 3D q-space views, hands a detector-frame crop to QoraPhase, and sorts
reconstructed amplitudes into true and twin groups.

Depends on `qorageom>=0.1.0`. Until QoraGeom is on PyPI, `[tool.uv.sources]` takes it from git,
and CXD cannot be uploaded before that.
