class X:
    pass


class Y:
    pass


class Z(X, Y):
    pass


class W(Y, X):
    pass


class V(Z, W):
    pass
