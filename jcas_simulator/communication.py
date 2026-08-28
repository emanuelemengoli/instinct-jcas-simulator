"""Communication queue model."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class LindleyQueue:
    arrival_rate: float
    service_scale: float = 1.0
    workload: float = 0.0
    workloads: list[float] = field(default_factory=lambda: [0.0])
    service_rates: list[float] = field(default_factory=list)

    def update(
        self,
        sinr: float,
        rng: np.random.Generator,
        *,
        service_enabled: bool = True,
    ) -> float:
        """Advance one queue slot.

        Arrivals occur every logical slot.  When a TDD phase disables
        communication, service is set to zero and no service draw is made.
        With ``service_enabled=True`` this is exactly the previous behavior.
        """
        if service_enabled:
            service_rate = self.service_scale * float(
                np.log2(1.0 + max(float(sinr), 0.0))
            )
        else:
            service_rate = 0.0
        arrivals = int(rng.poisson(self.arrival_rate))
        services = int(rng.poisson(max(service_rate, 0.0))) if service_enabled else 0
        self.workload = max(0.0, self.workload + arrivals - services)
        self.workloads.append(self.workload)
        self.service_rates.append(service_rate)
        return self.workload
