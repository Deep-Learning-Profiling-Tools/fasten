import torch


class StreamPool:
    _streams = []
    if torch.cuda.is_available():
        _streams.append(torch.cuda.current_stream())

    @classmethod
    def add(cls, nstreams: int = 1) -> None:
        for _ in range(nstreams):
            cls._streams.append(torch.cuda.Stream())

    @classmethod
    def reserve(cls, nstreams: int = 1) -> None:
        if torch.cuda.is_available():
            if len(cls._streams) < nstreams:
                cls.add(nstreams - len(cls._streams))

    @classmethod
    def get(cls, stream_idx: int = 1) -> torch.cuda.Stream:
        if torch.cuda.is_available():
            return cls._streams[stream_idx]
        else:
            return None

    @classmethod
    def size(cls) -> int:
        return len(cls._streams)
