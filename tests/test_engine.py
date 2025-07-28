import numpy as np
import pytest
from sentence_transformers import SentenceTransformer
from minivectordb.engine import MiniVectorDb

EPSILON = 1e-6

@pytest.mark.parametrize("text, expected_norm", [
    ("", 1.0),
    ("a", 1.0),
    ("abc", 1.0)
])
def test_string_to_embedding_norm(text, expected_norm):
    vector = MiniVectorDb.string_to_embedding(text)

    norm = np.linalg.norm(vector)
    print(norm)
    assert pytest.approx(norm, rel=EPSILON) == expected_norm


def test_string_to_embedding_repeat():
    text = "repeat"
    vector1 = MiniVectorDb.string_to_embedding(text)
    vector2 = MiniVectorDb.string_to_embedding(text)
    np.testing.assert_array_equal(vector1, vector2)


def test_add():
    db = MiniVectorDb()

    _id = "0"
    text = "Vector" + str(_id)

    db.add(_id, text)

    assert db.ids == [_id]
    assert db.texts[_id] == text


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
    db = MiniVectorDb()

    n = 2
    for i in range(n):
        _id = str(i)
        text = "Vector" + str(_id)
        db.add(_id, text)

    _id = str(0)
    db.delete(_id)

    assert db.vectors.shape[0] == n - 1

    assert _id not in db.ids
    assert _id not in db.texts.keys()


def test_cosine_similarity_identical():
    db = MiniVectorDb()

    _id = "0"
    text = "vector0"
    db.add(_id, text)
    vector = MiniVectorDb.string_to_embedding(text)

    similarities = db.cosine_similarity(vector)
    assert similarities.shape == (1,)
    assert pytest.approx(similarities[0], rel=EPSILON) == 1.0


def test_cosine_similarity_orthogonal():
    dim = 3
    db = MiniVectorDb(dim)

    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    db.add("x", x, "vector x")
    db.add("y", y, "vector y")

    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    similarities = db.cosine_similarity(z)

    assert similarities.shape == (2,)
    assert all(abs(s) < EPSILON for s in similarities)


def test_search_linear_2d():
    db = MiniVectorDb(dim=2)
    x = np.array([1, 0], dtype=np.float32)
    y = np.array([0, 1], dtype=np.float32)
    z = np.array([1, 1], dtype=np.float32)
    
    db.add("x", x, "vector x")
    db.add("y", y, "vector y")
    db.add("z", z, "vector z")
    
    query = np.array([1, 0], dtype=np.float32)
    results = db.search(query, k=3, method='linear')
    
    ids = [r[0] for r in results]
    similarities = [r[1] for r in results]
    texts = [r[2] for r in results]
    
    # x:  0 deg
    # z: 45 deg
    # y: 90 deg
    assert ids == ["x", "z", "y"]
    assert pytest.approx(similarities) == [1.0, 1 / np.sqrt(2), 0.0]
    assert texts == ["vector x", "vector z", "vector y"]


def test_search_linear_delete():
    db = MiniVectorDb(dim=2)
    x = np.array([1, 0], dtype=np.float32)
    y = np.array([0, 1], dtype=np.float32)
    
    db.add("x", x, "vector x")
    db.add("y", y, "vector y")

    db.delete("x")
    
    results = db.search(np.array([1, 0], dtype=np.float32), k=2, method='linear')
    ids = [r[0] for r in results]
    assert ids == ["y"]


def test_search_kd_tree_2d():
    db = MiniVectorDb(dim=2)
    x = np.array([1, 0], dtype=np.float32)
    y = np.array([0, 1], dtype=np.float32)
    z = np.array([1, 1], dtype=np.float32)
    
    db.add("x", x, "vector x")
    db.add("y", y, "vector y")
    db.add("z", z, "vector z")
    
    query = np.array([1, 0], dtype=np.float32)
    results = db.search(query, k=3, method='kdtree')
    
    ids = [r[0] for r in results]
    similarities = [r[1] for r in results]
    texts = [r[2] for r in results]
    
    # x:  0 deg
    # z: 45 deg
    # y: 90 deg
    assert ids == ["x", "z", "y"]
    assert pytest.approx(similarities) == [1.0, 1 / np.sqrt(2), 0.0]
    assert texts == ["vector x", "vector z", "vector y"]


def test_search_kd_tree_delete():
    db = MiniVectorDb(dim=2)
    x = np.array([1, 0], dtype=np.float32)
    y = np.array([0, 1], dtype=np.float32)
    
    db.add("x", x, "vector x")
    db.add("y", y, "vector y")

    db.delete("x")
    
    results = db.search(np.array([1, 0], dtype=np.float32), k=2, method='kdtree')
    ids = [r[0] for r in results]
    assert ids == ["y"]
