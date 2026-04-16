from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    benchmark_type: str
    machine_host: int
    platform_type: int
    gc_config: int
