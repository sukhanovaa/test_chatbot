# connect to containerized server
import requests
import signal
import time


host, port = '127.0.0.1', '6077'
address = f"http://{host}:{port}"


class Dialogue:
    __slots__ = ('session')

    def __init__(self):
        self.session = requests.Session()
        signal.signal(signal.SIGINT, self._on_exit)
        signal.signal(signal.SIGTERM, self._on_exit)

    def _on_exit(self, *args):
        print(self.session.get(f'{address}/metrics').json())
        self.session.get(f'{address}/exit').json()
        self.session.close()
        exit()

    def flow(self):
        while True:
            print(self.session.get(f'{address}/start').json())
            # start = time.time()
            user_input: str = input()   # TODO: timeout
            # self.user_times.append(time.time() - start)
            res = self.session.post(f'{address}/post', json={'text': user_input})
            print(res.json())


if __name__ == '__main__':
    d = Dialogue()
    d.flow()