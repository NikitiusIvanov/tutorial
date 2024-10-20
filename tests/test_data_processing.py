import main


def test_message_lenght():
    assert main.message_lenght(None) is None
    assert main.message_lenght('') == 0
    assert main.message_lenght('a') == 1