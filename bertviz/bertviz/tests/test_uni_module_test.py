from unittest import TestCase

from uni_module_test import FillDefaultDict


class TestFillDefaultDict(TestCase):
    def test_single_dependency(self):
        fdd = FillDefaultDict(a=lambda a: a, b=lambda a: a + 1, c=lambda a: a + 2)
        assert fdd(dict(a=1, c=10, r=7)) == dict(a=1, b=2, c=10)
        assert fdd(2) == dict(a=2, b=3, c=4)

    def test_circular_dependency(self):
        fdd = FillDefaultDict(
            a=lambda b, c, r: b * c + r,
            b=lambda a, c, r: (a - r) // c,
            c=lambda a, b: a // b,
            r=lambda a, b: a % b,
        )
        assert fdd(dict(a=7, b=3, c=9)) == fdd(dict(a=7, b=3, c=9, r=1))
        assert fdd(dict(b=9, c=2, r=3)) == fdd(dict(a=21, b=9, c=2, r=3))
        try:
            fdd(dict(a=1))
            self.fail()
        except ValueError:
            pass

    def test_multi_pass(self):
        fdd = FillDefaultDict(
            cnt=lambda sum, mean: int(0.5+sum / mean),
            mean=lambda cnt, sum: sum / cnt,
            sum=lambda cnt, mean: mean * cnt,
            sum_2=lambda std, mean, cnt: (std + mean ** 2) * cnt,
            mean_2=lambda sum_2, cnt: sum_2 / cnt,
            std=lambda mean, mean_2: mean_2 - mean ** 2,
        )
        assert fdd(
            dict(cnt=5, sum=10, std=0)
        ) == dict(
            cnt=5, mean=2, sum=10, sum_2=20, mean_2=4, std=0
        )
