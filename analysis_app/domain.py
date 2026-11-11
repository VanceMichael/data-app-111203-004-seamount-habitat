import math
import statistics
from collections import Counter


def describe(records):
    return {'represented_area': math.fsum(row['weight'] for row in records), 'substrate_counts': dict(sorted(Counter(row['substrate'] or '未知' for row in records).items())), 'surveys': len({row['survey'] for row in records})}
