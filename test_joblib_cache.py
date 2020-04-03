import joblib

mem = joblib.Memory("cache")

@mem.cache
def f(x):
    return g(x) + 4

def g(x):
    return x * 2

if __name__ == "__main__":
    print(f(3), f(9))