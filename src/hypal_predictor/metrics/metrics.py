from dataclasses import dataclass


@dataclass
class RegressionMetrics:
    r2: float = 0
    mse: float = 0
    mae: float = 0


@dataclass
class ClassificationMetrics:
    critical_detection_precision: float = 0
    critical_detection_recall: float = 0
    critical_detection_f1: float = 0
    critical_undetection_precision: float = 0
    critical_undetection_recall: float = 0
    critical_undetection_f1: float = 0

    def f2(self, k: float = 0.5) -> float:
        assert k >= 0 and k <= 1, "k must be between 0 and 1"
        return k * self.critical_detection_f1 + (1 - k) * self.critical_undetection_f1

    @property
    def f2_025(self) -> float:
        return self.f2(k=0.25)

    @property
    def f2_050(self) -> float:
        return self.f2(k=0.50)

    @property
    def f2_075(self) -> float:
        return self.f2(k=0.75)


@dataclass
class Metrics:
    regression: RegressionMetrics
    classification: ClassificationMetrics
