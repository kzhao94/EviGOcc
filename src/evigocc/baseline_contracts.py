from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationRef:
    index: int
    sequence: str
    timestamp: int


@dataclass(frozen=True)
class CausalBlock:
    sequence: str
    start: int
    frames: tuple[int, ...]
    context: tuple[int, ...]


def causal_blocks(
    observations: list[ObservationRef],
    *,
    block_size: int,
    context_size: int,
) -> list[CausalBlock]:
    """Partition observations without future-frame context or sequence crossing."""
    if block_size < 1 or context_size < 0:
        raise ValueError("block_size must be positive and context_size nonnegative")
    grouped: dict[str, list[ObservationRef]] = {}
    for observation in observations:
        grouped.setdefault(observation.sequence, []).append(observation)
    output = []
    for sequence in sorted(grouped):
        ordered = sorted(grouped[sequence], key=lambda item: (item.timestamp, item.index))
        if len({item.index for item in ordered}) != len(ordered):
            raise ValueError(f"duplicate observation index in {sequence}")
        for start in range(0, len(ordered), block_size):
            context_start = max(0, start - context_size)
            output.append(
                CausalBlock(
                    sequence=sequence,
                    start=start,
                    frames=tuple(item.index for item in ordered[start : start + block_size]),
                    context=tuple(item.index for item in ordered[context_start:start]),
                )
            )
    return output


def ttocc_contract(observations: list[ObservationRef]) -> list[CausalBlock]:
    return causal_blocks(observations, block_size=16, context_size=3)


def freeocc_contract(observations: list[ObservationRef]) -> list[CausalBlock]:
    return causal_blocks(observations, block_size=64, context_size=16)
