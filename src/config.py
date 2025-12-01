#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    benchmark_type: str
    machine_host: int
    platform_type: int
    gc_config: int

if __name__ == "__main__":
    pass
