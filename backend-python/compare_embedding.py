import requests
from scipy.spatial.distance import cosine
import numpy as np


def get_embedding(texts=['xin chào']):
    with requests.post(
        url='http://127.0.0.1:8000/embeddings',
        json={
            "input": texts,
            "model": "rwkv",
            "encoding_format": "utf8",
            "fast_mode": False
        }
    ) as rs:
        return [i['embedding'] for i in rs.json()['data']]


def distance(text1, text2):
    embeddings = np.array(get_embedding([text1, text2]))
    # print(embeddings.shape)
    # print(embeddings)
    score = 1 - cosine(embeddings[0], embeddings[1])
    print(score)


if __name__ == '__main__':
    distance(
        "đăng ký gói cước st5k ",
        "sao không đăng ký được gói cước st5k ",
    )

    distance(
        "đăng ký gói cước st5k ",
        "đăng ký gói cước st120k ",
    )

    distance(
        "đăng ký gói cước st5k ",
        "hủy gói cước st120k ",
    )

    distance(
        "đăng ký gói cước st5k ",
        "cài đặt gói cước st120k ",
    )

    distance(
        "đăng ký gói cước st5k ",
        "cài đặt gói cước st5k ",
    )