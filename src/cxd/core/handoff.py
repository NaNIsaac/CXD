"""Hand a detector-frame rocking stack to QoraPhase.

QoraPhase takes a centre-origin intensity (index ``n // 2`` is q=0) and a mask where
``True`` is an unusable pixel. It rejects NaN/Inf even at masked pixels, so those are
zero-filled here. Note that ``cxd.mapping.qspace`` uses the opposite polarity
(``True`` = kept).

The stack stays in detector frame, ``(frame, y, x)``, without interpolation: no
q-space regridding happens here, so the reciprocal-space basis of the cropped
voxels has to travel with it separately.
"""

import numpy as np


def to_qoraphase(
    stack: np.ndarray,
    size: int | tuple[int, int, int],
    bad: np.ndarray | None = None,
    center: tuple[int, int, int] | None = None,
    frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Crop ``stack`` around the Bragg peak into QoraPhase's input arrays.

    stack: ``(frame, y, x)`` intensity in counts
    size: crop size, one int for a cube or one per axis
    bad: ``True`` where a pixel is unusable (gaps, hot, saturated)
    center: peak position; defaults to the brightest valid voxel
    frames: number of rocking steps in the scan; if given, axis 0 must have this length

    Returns ``(intensity, mask, offset)``: float64 intensity with the peak at
    ``n // 2`` of each axis and unusable voxels set to 0, the mask (``True`` =
    unusable; also NaN, Inf and negative voxels), and the index of the crop's
    first voxel in ``stack``.
    """
    stack = np.asarray(stack, dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError(f"stack must be (frame, y, x); got shape {stack.shape}")
    if frames is not None and stack.shape[0] != frames:
        raise ValueError(
            f"stack axis 0 has {stack.shape[0]} entries but the scan has {frames} frames; "
            "is it (y, x, frame), as in the .mat files?"
        )
    shape = (size,) * 3 if isinstance(size, int) else tuple(size)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"size must be a positive int or three of them; got {size}")

    invalid = ~np.isfinite(stack) | (stack < 0)
    if bad is not None:
        if bad.shape != stack.shape:
            raise ValueError(f"bad has shape {bad.shape}, stack {stack.shape}")
        invalid |= bad.astype(bool)
    if invalid.all():
        raise ValueError("every voxel is unusable")

    peak = (
        center
        if center is not None
        else np.unravel_index(np.argmax(np.where(invalid, -np.inf, stack)), stack.shape)
    )
    offset = tuple(int(c) - n // 2 for c, n in zip(peak, shape, strict=True))
    if any(o < 0 or o + n > s for o, n, s in zip(offset, shape, stack.shape, strict=True)):
        raise ValueError(
            f"a {shape} crop centred on {tuple(int(c) for c in peak)} runs off "
            f"the {stack.shape} stack; use a smaller size"
        )

    crop = tuple(slice(o, o + n) for o, n in zip(offset, shape, strict=True))
    mask = invalid[crop]
    # cxd.mapping.qspace masks are True = kept; passed here they mark most voxels bad
    if mask.mean() > 0.5:
        raise ValueError(
            f"{mask.mean():.0%} of the crop is marked unusable; check the mask "
            "polarity (here True = unusable)"
        )
    return np.where(mask, 0.0, stack[crop]), mask, offset
