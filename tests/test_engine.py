import numpy as np
import pytest
from minivectordb.engine import MiniVectorDb

@pytest.mark.parametrize("text, dim, expected_norm", [
    ("", 10, 0.0),
    ("a", 10, 1.0),
    ("aa", 10, 1.0),
    ("ab", 10, 1.0)
])
def test_string_to_embedding_norm(text, dim, expected_norm):
    vector = MiniVectorDb.string_to_embedding(text, dim)

    norm = np.linalg.norm(vector)
    assert pytest.approx(norm, rel=1e-6) == expected_norm


def test_string_to_embedding_repeat():
    text = "repeat"
    dim = 128
    vector1 = MiniVectorDb.string_to_embedding(text, dim)
    vector2 = MiniVectorDb.string_to_embedding(text, dim)
    np.testing.assert_array_equal(vector1, vector2)


def test_add():
    dim = 4
    db = MiniVectorDb(dim)

    vector = np.array(np.linspace(0.0, float(dim), dim), dtype=np.float32)
    _id = "0"
    text = "Vector" + str(_id)

    db.add(_id, vector, text)

    assert db.vectors.shape == (1, dim)
    assert db.ids == [_id]
    assert db.texts[_id] == text
    np.testing.assert_array_equal(db.vectors[0], vector)


def test_add_multiple():
    dim = 8
    db = MiniVectorDb(dim)

    n = 10
    for i in range(n):
        vector = np.array(np.linspace(float(i), float(dim + i), dim), dtype=np.float32)
        _id = str(i)
        text = "Vector" + str(_id)
        db.add(_id, vector, text)

    assert db.vectors.shape == (n, dim)

    for i in range(n):
        vector = np.array(np.linspace(float(i), float(dim + i), dim), dtype=np.float32)
        _id = str(i)
        text = "Vector" + str(_id)

        assert db.ids[i] == _id
        assert db.texts[_id] == text


def test_delete():
    dim = 8
    db = MiniVectorDb(dim)

    n = 2
    for i in range(n):
        vector = np.array(np.linspace(float(i), float(dim + i), dim), dtype=np.float32)
        _id = str(i)
        text = "Vector" + str(_id)
        db.add(_id, vector, text)

    _id = str(0)
    text = "Vector" + str(_id)
    db.delete(_id)

    assert db.vectors.shape == (n - 1, dim)

    assert _id not in db.ids
    assert _id not in db.texts.keys()
