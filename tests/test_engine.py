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

    key = "0"
    text = "Vector" + str(key)

    db.add(key, text)

    assert db.keys == [key]
    assert db.texts[key] == text


def test_add_multiple():
    db = MiniVectorDb()

    n = 10
    for i in range(n):
        key = str(i)
        text = "Vector" + str(key)
        db.add(key, text)

    assert db.vectors.shape[0] == n

    for i in range(n):
        key = str(i)
        text = "Vector" + str(key)

        assert db.keys[i] == key
        assert db.texts[key] == text

def test_add_batch():
    db = MiniVectorDb()

    n = 256
    keys = [str(i) for i in range(n)]
    texts = ["Vector" + str(i) for i in range(n)]

    db.add_batch(keys, texts, n)
    db.add_batch(keys, texts, n)

    assert db.vectors.shape[0] == 2 * n

    for i in range(n):
        key = str(i)
        text = "Vector" + str(key)

        assert db.keys[i] == key
        assert db.texts[key] == text


def test_delete():
    db = MiniVectorDb()

    n = 2
    for i in range(n):
        key = str(i)
        text = "Vector" + str(key)
        db.add(key, text)

    key = str(0)
    db.delete(key)

    assert db.vectors.shape[0] == n - 1

    assert key not in db.keys
    assert key not in db.texts.keys()


def test_delete_empty():
    db = MiniVectorDb()

    key = str(0)
    db.delete(key)

    assert db.vectors.shape[0] == 0

    assert db.keys == []
    assert len(db.texts.keys()) == 0


def test_cosine_similarity_identical():
    db = MiniVectorDb()

    key = "0"
    text = "vector0"
    db.add(key, text)
    vector = MiniVectorDb.string_to_embedding(text)

    similarities = db.cosine_similarity(vector)
    assert similarities.shape == (1,)
    assert pytest.approx(similarities[0], rel=EPSILON) == 1.0


def test_search_linear_2d():
    db = MiniVectorDb()

    text1 = "The sky is blue."
    text2 = "A cat with a hat."
    text3 = "The sky is light blue."
    
    db.add("1", text1)
    db.add("2", text2)
    db.add("3", text3)
    
    results = db.search(text1, k=3, method='linear')
    
    keys = [r[0] for r in results]
    similarities = [r[1] for r in results]
    texts = [r[2] for r in results]
    
    assert keys == ["1", "3", "2"]
    assert texts == [text1, text3, text2]


def test_search_linear_delete():
    db = MiniVectorDb()

    text1 = "The sky is blue."
    text2 = "A cat with a hat."
    
    db.add("1", text1)
    db.add("2", text2)

    db.delete("1")
    
    results = db.search(text1, k=2, method='linear')
    keys = [r[0] for r in results]
    assert keys == ["2"]


def test_search_kd_tree_2d():
    db = MiniVectorDb()

    text1 = "The sky is blue."
    text2 = "A cat with a hat."
    text3 = "The sky is light blue."
    
    db.add("1", text1)
    db.add("2", text2)
    db.add("3", text3)
    
    results = db.search(text1, k=3, method='kdtree')
    
    keys = [r[0] for r in results]
    similarities = [r[1] for r in results]
    texts = [r[2] for r in results]
    
    assert keys == ["1", "3", "2"]
    assert texts == [text1, text3, text2]


def test_search_kd_tree_delete():
    db = MiniVectorDb()

    text1 = "The sky is blue."
    text2 = "A cat with a hat."
    
    db.add("1", text1)
    db.add("2", text2)

    db.delete("1")
    
    results = db.search(text1, k=2, method='kdtree')
    keys = [r[0] for r in results]
    assert keys == ["2"]

def test_rebuild_ivf_empty():
    db = MiniVectorDb()
    db.rebuild_ivf()
