import threading

counter = 0

def f():
    global counter
    for _ in range(100000):
        counter = counter + 1


t1 = threading.Thread(target=f)
t2 = threading.Thread(target=f)

t1.start()
t2.start()

t1.join()
t2.join()

print(counter)
