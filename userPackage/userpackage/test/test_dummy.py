import pytest
from userpackage.utils.dummy import dummy_prog

def test_dummy_output(capsys):
    # Test for printing numbers from 1 to 8
    dummy_prog(8)
    captured = capsys.readouterr()
    assert captured.out == "1 2 3 4 5 6 7 8 \n"

    # Test for printing numbers from 1 to 10
    dummy_prog(10)
    captured = capsys.readouterr()
    assert captured.out == "1 2 3 4 5 6 7 8 9 10 \n"

    # Test for printing numbers from 1 to 1
    dummy_prog(1)
    captured = capsys.readouterr()
    assert captured.out == "1 \n"

    # Test for edge case: n = 0 (although this might not be useful, depends on your constraints)
    dummy_prog(0)
    captured = capsys.readouterr()
    assert captured.out == "\n"