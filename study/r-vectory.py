import numpy as np
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()


B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(type(B))

nr, nc = B.shape
Br = ro.r.matrix(B, nrow=nr, ncol=nc)

print(type(Br))
ro.r.assign("B", Br)
