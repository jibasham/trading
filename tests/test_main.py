from trading.main import main


def test_main_prints_hello(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from trading!" in captured.out
