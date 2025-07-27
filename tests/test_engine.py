import numpy as np

from minivectordb.engine import MiniVectorDb


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
