def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests under Testing/ (requires torch/transformers/peft + model setup).",
    )


def pytest_ignore_collect(collection_path, config):
    # Skip integration tests unless explicitly enabled
    if "Testing" in str(collection_path) and not config.getoption("--run-integration"):
        return True
    return False
