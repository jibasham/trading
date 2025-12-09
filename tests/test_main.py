from pathlib import Path
import sys

# Ensure project root is on sys.path so the 'trading' package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading.main import main


def test_main_prints_hello(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from trading!" in captured.out
