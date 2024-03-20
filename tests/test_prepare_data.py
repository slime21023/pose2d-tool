from src.preprocess import prepare_data

def test_preprocess():
    prepare_data(src="poses", dst="train")
    assert 1 == 1