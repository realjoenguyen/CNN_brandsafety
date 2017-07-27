from multiprocessing import Lock, Semaphore, Array


class Pool(object):
    """Create a pool of object with the given class_type, args, and kwargs.

    The get method will return any objects currently unused from the pool.
    The put method will put back the object, marking it as unused for future use.

    Putting blocking=True will wait until an object is available, while blocking=False will return immediately if
    there is no object available
    """

    def __init__(self, class_type, n, blocking=True, callback=lambda x: x, args=(), kwargs={}):
        self.class_type = class_type
        self.items = []
        self.used = Array('i', [0] * n)
        self.lock = Lock()
        self.count = Semaphore(n)
        self.n = n
        self.blocking = blocking
        try:
            for i in range(n):
                my_item = self.class_type(*args, **kwargs)
                callback(my_item)
                self.items.append(my_item)
        except Exception as exc:
            raise type(exc)('%s: %s' % (self.class_type, exc.message))

    def get(self):
        self.lock.acquire(self.blocking)
        try:
            if not self.count.acquire(self.blocking):
                return None
            for i, my_item in enumerate(self.items):
                if self.used[i] == 0:
                    self.used[i] = 1
                    return my_item
        finally:
            self.lock.release()

    def put(self, item):
        self.lock.acquire()
        try:
            for i, my_item in enumerate(self.items):
                if my_item == item:
                    self.used[i] = 0
                    self.count.release()
                    return True
            return False
        finally:
            self.lock.release()
