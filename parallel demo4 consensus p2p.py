from cvxpy import *
import numpy as np
import scipy.sparse as sps
from multiprocessing import Process, Pipe
import matplotlib.pyplot as plt

def buildCommpipes(pipeID):
    # pipeStart là mảng chứa các đầu vào của pipe
    pipeStart = []
    # pipeEnd là mảng chứa các đầu ra của pipe
    pipeEnd = []
    # Vòng lặp chạy từ 0 đến số hàng của pipeID
    for i in range(len(pipeID)):
        # start, end là 2 đầu của pipe
        start, end = Pipe()
        # pipeStart là mảng chứa các đầu vào của pipe
        pipeStart += [start]
        # pipeEnd là mảng chứa các đầu ra của pipe
        pipeEnd += [end]
    return pipeStart, pipeEnd

def runWorkerWithPlot(xlocal, pipeID, pipeStart, pipeEnd, workerID, plot_values):
    # xlocal là giá trị x ban đầu của worker
    nvar = len(xlocal)
    # Khơi tạo các biến
    rho = 1
    epsilon = 0.001
    # xbar là biến xbar của worker
    # xbar = np.zeros(nvar) = [0,0,0,0]
    xbar = Parameter(nvar, value=np.zeros(nvar))
    # u = [0,0,0,0]
    u = Parameter(nvar, value=np.zeros(nvar))
    # x = [0,0,0,0] -> biến cần tối ưu
    x = Variable(nvar)
    # f = ||x - xlocal||^2 + 0.5*rho*||x - xbar + u||^2
    f = square(norm(x - xlocal)) + (rho/2)*sum_squares(x - xbar + u)
    # tìm minium của f
    prox = Problem(Minimize(f))
    # pipeLocal là mảng chứa các pipe của worker
    pipeLocal = []
    # destLocal là mảng chứa các worker kết nối với worker hiện tại
    destLocal = []
    # Vòng lặp chạy từ 0 đến số hàng của pipeID
    for i in range(len(pipeID)):
        # Nếu cột 1 của pipeID bằng workerID thì thêm pipeStart[i] vào pipeLocal
        if pipeID[i, 1] == workerID:
            pipeLocal += [pipeStart[i]]
            destLocal += pipeID[i, 2]
        # Nếu cột 2 của pipeID bằng workerID thì thêm pipeEnd[i] vào pipeLocal
        if pipeID[i, 2] == workerID:
            pipeLocal += [pipeEnd[i]]
            destLocal += pipeID[i, 1]
    # nPipe là số pipe của worker
    nPipe = len(pipeLocal)
    # N là số worker
    N = nPipe + 1
    k = 0
    # convg là biến kiểm tra điều kiện dừng
    convg = False
    # x_values là mảng chứa các giá trị x tại mỗi vòng lặp
    x_values = []
    # Tìm giá trị x ban đầu
    prox.solve()
    # Thêm giá trị x ban đầu vào x_values
    msgrcv = np.tile(np.array(x.value.transpose()), (nPipe, 1)).transpose()
    print(msgrcv)
    while not convg:
        k += 1
        prox.solve()
        x_values.append(x.value.copy())



        for i in range(nPipe):
            # Gửi giá trị x tới các worker kết nối với worker hiện tại
            msg = x.value
            pipeLocal[i].send(msg)

            # Nhận giá trị x từ các worker kết nối với worker hiện tại
            if pipeLocal[i].poll(0.0001):
                msgrcv[:, i] = pipeLocal[i].recv().transpose()

        # Tính giá trị xbar, Trung bình của giá trị x từ tất cả các workers.
        xbar.value = (np.sum(msgrcv, axis=1) + x.value.transpose()).transpose() / N

        # Tính giá trị u
        u.value += x.value - xbar.value

        if np.linalg.norm(xbar.value - x.value) < epsilon and k > 5:
            convg = True

    if convg:
        print(f"Worker {workerID} converged at {k} -th iteration! The result is")
        print(x.value)

        if plot_values:
            # Plot the values of x over iterations
            plt.plot(range(1, k+1), x_values)
            plt.xlabel('Iteration')
            plt.ylabel(f'x values - Worker {workerID}')
            plt.title(f'Convergence Plot - Worker {workerID}')
            plt.show()

if __name__ == '__main__':
    # Define the problem data
    # nvar là số chiềuz
    nvar = 4
    # N là số worker
    N = 3
    # A là ma trận kề lưu đồ thị lưu thông của các worker
    A = sps.csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    print("A", A.toarray())
    # Chuyển ma trâận kề A thành ma trận trên Atr
    Atr = sps.triu(A)
    print("Atr",Atr.toarray())
    # Lấy ra các giá trị khác 0 của ma trận Atr
    # r là chỉ số hàng, c là chỉ số cột, v là giá trị
    r, c, v = sps.find(Atr)
    print(np.array([r,c,v]))
    # pipeID là ma trận 3 cột, cột đầu là ID của kết nối, cột 2 là chỉ số hàng, cột 3 là chỉ số cột
    pipeID = np.array([np.arange(r.size), r, c]).transpose()
    print("pipeID", pipeID)
    pipeStart, pipeEnd = buildCommpipes(pipeID)
    np.random.seed(0)
    # Sinh ngâu nhiên 1 ma trận a (nvar x N) -> (4x3)
    a = np.random.randn(nvar, N)
    print("initial values of x's are \n", a)

    procs = []
    for i in range(N):
         procs += [Process(target=runWorkerWithPlot, args=(a[:, i], pipeID, pipeStart, pipeEnd, i, True))]
         procs[-1].start()

    for proc in procs:
         proc.join()
