from functools import partial

def xx(x):
    return x

def yy(y):
    return y

if __name__ == '__main__':
    x = [partial(xx, 1), partial(yy, 3)]
    y = x[0]()
    print(y)
