import random
from core.utils import DataSource

VALUES = [-1.0, 1.0]

class SecretGenerator(DataSource):
    def _generate_one(self):
        return [float(VALUES[random.randint(0, 1)]) for _ in range(self.dim)]

class MessageGenerator(DataSource):
    def _generate_one(self):
        return [float(VALUES[random.randint(0, 1)]) for _ in range(self.dim)]