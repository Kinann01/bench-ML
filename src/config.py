#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Config:

    def __init__(
        self,
        benchmark_type: str,
        machine_host: int,
        platform_type: int,
        gc_config: int,
    ):
        self.benchmark_type = benchmark_type
        self.machine_host = machine_host
        self.platform_type = platform_type
        self.gc_config = gc_config

    def __repr__(self) -> str:
        return (
            f"Config(benchmark_type='{self.benchmark_type}', "
            f"machine_host='{self.machine_host}', "
            f"platform_type={self.platform_type}, "
            f"gc_config={self.gc_config})"
        )

if __name__ == "__main__":
    pass
