# connect to containerized server
import requests, yaml
from utils import ExitHandler

config = yaml.load('src/config.yaml')
host, port = '127.0.0.1', config['client_port']
address = f"http://{host}:{port}"


def flow():
    exit_handler = ExitHandler()
    with requests.Session() as sess:
        print(sess.get(f'{address}/start'))
        while not exit_handler.exiting:
            # start = time.time()
            user_input: str = input()   # TODO: timeout
            # self.user_times.append(time.time() - start)
            res = sess.post(f'{address}/post', user_input)
            print(res)
        else:
            print(sess.get(f'{address}/metrics'))
            exit()


if __name__ == '__main__':
    flow()